# python lib
import os, sys, argparse, glob, re, math, copy, pickle
import time
from datetime import datetime
import itertools
import numpy as np

# torch & torch lib
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.autograd as autograd
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
from tensorboardX import SummaryWriter

# other files
import datasets
import utils
import networks

def Pre_train(opts):
    ### initialize model
    print('===> Initializing model from %s...' %opts.model)
    model = networks.__dict__[opts.model](opts, nc_in=12, nc_out=3)
    ### initialize optimizer
    if opts.solver == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=opts.lr_init, momentum=opts.momentum, weight_decay=opts.weight_decay)
    elif opts.solver == 'ADAM':
        optimizer = optim.Adam(model.parameters(), lr=opts.lr_init, weight_decay=opts.weight_decay, betas=(opts.beta1, opts.beta2))
    else:
        raise Exception("Not supported solver (%s)" %opts.solver)


    ### resume latest model
    name_list = glob.glob(os.path.join(opts.model_dir, "model_epoch_*.pth"))
    epoch_st = 0
    if len(name_list) > 0:
        epoch_list = []
        for name in name_list:
            s = re.findall(r'\d+', os.path.basename(name))[0]
            epoch_list.append(int(s))

        epoch_list.sort()
        epoch_st = epoch_list[-1]


    if epoch_st > 0:

        print('=====================================================================')
        print('===> Resuming model from epoch %d' %epoch_st)
        print('=====================================================================')

        ### resume latest model and solver
        model, optimizer = utils.load_model(model, optimizer, opts, epoch_st)

    else:
        ### save epoch 0
        utils.save_model(model, optimizer, opts)


    print(model)

    num_params = utils.count_network_parameters(model)

    print('\n=====================================================================')
    print("===> Model has %d parameters" %num_params)
    print('=====================================================================')


    ### initialize loss writer
    loss_dir = os.path.join(opts.model_dir, 'loss')
    loss_writer = SummaryWriter(loss_dir)
    
    

    ### Load pretrained FlowNet2
    opts.rgb_max = 1.0
    opts.fp16 = False

    FlowNet = utils.create_pwcnet(opts)

    ### Load pretrained VGG
    VGG = networks.Vgg16(requires_grad=False)

    ### convert to GPU
    device = torch.device("cuda" if opts.cuda else "cpu")

    model = model.to(device)
    FlowNet = FlowNet.to(device)
    VGG = VGG.to(device)
    
    model.train()
    
    ### create dataset
    train_dataset = datasets.MultiFramesDataset(opts, "train")
    
    ### start training
    while model.epoch < opts.epoch_max:

        model.epoch += 1

        ### re-generate train data loader for every epoch
        data_loader = utils.create_data_loader(train_dataset, opts, "train")

        ### update learning rate
        current_lr = utils.learning_rate_decay(opts, model.epoch)

        for param_group in optimizer.param_groups:
            param_group['lr'] = current_lr

        ### criterion and loss recorder
        if opts.loss == 'L2':
            criterion = nn.MSELoss(size_average=True)
        elif opts.loss == 'L1':
            criterion = nn.L1Loss(size_average=True)
        else:
            raise Exception("Unsupported criterion %s" %opts.loss)
        

        ### start epoch
        ts = datetime.now()
        
        for iteration, batch in enumerate(data_loader, 1):
            
            total_iter = (model.epoch - 1) * opts.train_epoch_size + iteration

            ### convert data to cuda
            frame_i = []
            frame_p = []
            for t in range(opts.sample_frames):
                frame_i.append(batch[t * 2].to(device))
                frame_p.append(batch[t * 2 + 1].to(device))

            frame_o = []
            frame_o.append(frame_p[0]) ## first frame

            ### get batch time
            data_time = datetime.now() - ts

            ts = datetime.now()

            ### clear gradients
            optimizer.zero_grad()

            lstm_state = None
            ST_loss = 0
            LT_loss = 0
            VGG_loss = 0

            ### forward
            for t in range(1, opts.sample_frames):

                frame_i1 = frame_i[t - 1]
                frame_i2 = frame_i[t]
                frame_p2 = frame_p[t]

                if t == 1:
                    frame_o1 = frame_p[t - 1]
                else: 
                    frame_o1 = frame_o2.detach()    ## previous output frame

                frame_o1.requires_grad = False 

                ### model input        
                inputs = torch.cat((frame_p2, frame_o1, frame_i2, frame_i1), dim=1)
                
                ### forward model
                output, lstm_state = model(inputs, lstm_state)

                ### residual learning
                frame_o2 = output + frame_p2
                ## detach from graph and avoid memory accumulation
                lstm_state = utils.repackage_hidden(lstm_state)

                frame_o.append(frame_o2)

                ### short-term temporal loss
                if opts.w_ST > 0:
                
                    ### compute flow (from I2 to I1)
                    flow_i21 = networks.PWCEstimate(FlowNet ,frame_i2, frame_i1)

                    ### warp I1 and O1

                    warp_i1 = networks.PWCNetBackward(frame_i1, flow_i21)
                    warp_o1 = networks.PWCNetBackward(frame_o1, flow_i21)

                    ### compute non-occlusion mask: exp(-alpha * || F_i2 - Warp(F_i1) ||^2 )
                    noc_mask2 = torch.exp( -opts.alpha * torch.sum(frame_i2 - warp_i1, dim=1).pow(2) ).unsqueeze(1)
                    ST_loss += opts.w_ST * criterion(frame_o2 * noc_mask2, warp_o1 * noc_mask2)

                ### perceptual loss
                if opts.w_VGG > 0:

                    ### normalize
                    frame_o2_n = utils.normalize_ImageNet_stats(frame_o2)
                    frame_p2_n = utils.normalize_ImageNet_stats(frame_p2)
                    
                    ### extract VGG features
                    features_p2 = VGG(frame_p2_n, opts.VGGLayers[-1])
                    features_o2 = VGG(frame_o2_n, opts.VGGLayers[-1])

                    VGG_loss_all = []
                    for l in opts.VGGLayers:
                        VGG_loss_all.append( criterion(features_o2[l], features_p2[l]) )
                        
                    VGG_loss += opts.w_VGG * sum(VGG_loss_all)

            ## end of forward


            ### long-term temporal loss
            if opts.w_LT > 0:

                t1 = 0
                for t2 in range(t1 + 2, opts.sample_frames):

                    frame_i1 = frame_i[t1]
                    frame_i2 = frame_i[t2]

                    frame_o1 = frame_o[t1].detach() ## make a new Variable to avoid backwarding gradient
                    frame_o1.requires_grad = False

                    frame_o2 = frame_o[t2]

                    ### compute flow (from I2 to I1)
                    flow_i21 = networks.PWCEstimate(FlowNet, frame_i2, frame_i1)
                    
                    ### warp I1 and O1
                    warp_i1 = networks.PWCNetBackward(frame_i1, flow_i21)
                    warp_o1 = networks.PWCNetBackward(frame_o1, flow_i21)

                    ### compute non-occlusion mask: exp(-alpha * || F_i2 - Warp(F_i1) ||^2 )
                    noc_mask2 = torch.exp( -opts.alpha * torch.sum(frame_i2 - warp_i1, dim=1).pow(2) ).unsqueeze(1)

                    LT_loss += opts.w_LT * criterion(frame_o2 * noc_mask2, warp_o1 * noc_mask2)

                ### end of t2
            ### end of w_LT
            

            ### overall loss
            overall_loss = ST_loss + LT_loss + VGG_loss

            ### backward loss
            overall_loss.backward()

            ### update parameters
            optimizer.step()
                
            network_time = datetime.now() - ts


            ### print training info
            info = "[GPU %d]: " %(opts.gpu)
            info += "Epoch %d; Batch %d / %d; " %(model.epoch, iteration, len(data_loader))
            info += "lr = %s; " %(str(current_lr))

            ## number of samples per second
            batch_freq = opts.batch_size / (data_time.total_seconds() + network_time.total_seconds())
            info += "data loading = %.3f sec, network = %.3f sec, batch = %.3f Hz\n" %(data_time.total_seconds(), network_time.total_seconds(), batch_freq)
            
            info += "\tmodel = %s\n" %opts.model_name

            ### print and record loss
            if opts.w_ST > 0:
                loss_writer.add_scalar('ST_loss', ST_loss.item(), total_iter)
                info += "\t\t%25s = %f\n" %("ST_loss", ST_loss.item())

            if opts.w_LT > 0:
                loss_writer.add_scalar('LT_loss', LT_loss.item(), total_iter)
                info += "\t\t%25s = %f\n" %("LT_loss", LT_loss.item())

            if opts.w_VGG > 0:
                loss_writer.add_scalar('VGG_loss', VGG_loss.item(), total_iter)
                info += "\t\t%25s = %f\n" %("VGG_loss", VGG_loss.item())

            loss_writer.add_scalar('Overall_loss', overall_loss.item(), total_iter)
            info += "\t\t%25s = %f\n" %("Overall_loss", overall_loss.item())

            print(info)

        ### end of epoch

        ### save model
        utils.save_model(model, optimizer, opts)