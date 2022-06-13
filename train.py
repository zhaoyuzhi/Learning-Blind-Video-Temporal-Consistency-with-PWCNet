import argparse
import os
import torch

import trainer as trainer

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fast Blind Video Temporal Consistency")

    ### model options
    parser.add_argument('-model',           type=str,     default="TransformNet",   help='TransformNet') 
    parser.add_argument('-nf',              type=int,     default=32,               help='#Channels in conv layer')
    parser.add_argument('-blocks',          type=int,     default=5,                help='#ResBlocks') 
    parser.add_argument('-norm',            type=str,     default='IN',             choices=["BN", "IN", "none"],   help='normalization layer')
    parser.add_argument('-model_name',      type=str,     default='none',           help='path to save model')

    ### dataset options
    parser.add_argument('-datasets_tasks',  type=str,     default='W3_D1_C1_I1',    help='dataset-task pairs list')
    parser.add_argument('-data_dir',        type=str,     default='data',           help='path to data folder')
    parser.add_argument('-list_dir',        type=str,     default='lists',          help='path to lists folder')
    parser.add_argument('-checkpoint_dir',  type=str,     default='checkpoints',    help='path to checkpoint folder')
    parser.add_argument('-crop_size',       type=int,     default=192,              help='patch size')
    parser.add_argument('-geometry_aug',    type=int,     default=1,                help='geometry augmentation (rotation, scaling, flipping)')
    parser.add_argument('-order_aug',       type=int,     default=1,                help='temporal ordering augmentation')
    parser.add_argument('-scale_min',       type=float,   default=0.5,              help='min scaling factor')
    parser.add_argument('-scale_max',       type=float,   default=2.0,              help='max scaling factor')
    parser.add_argument('-sample_frames',   type=int,     default=11,               help='#frames for training')
        
    ### loss optinos
    parser.add_argument('-alpha',           type=float,   default=50.0,             help='alpha for computing visibility mask')
    parser.add_argument('-loss',            type=str,     default="L1",             help="optimizer [Options: SGD, ADAM]")
    parser.add_argument('-w_ST',            type=float,   default=100,              help='weight for short-term temporal loss')
    parser.add_argument('-w_LT',            type=float,   default=100,              help='weight for long-term temporal loss')
    parser.add_argument('-w_VGG',           type=float,   default=10,               help='weight for VGG perceptual loss')
    parser.add_argument('-VGGLayers',       type=str,     default="4",              help="VGG layers for perceptual loss, combinations of 1, 2, 3, 4")
                

    ### training options
    parser.add_argument('-solver',          type=str,     default="ADAM",           choices=["SGD", "ADAIM"],   help="optimizer")
    parser.add_argument('-momentum',        type=float,   default=0.9,              help='momentum for SGD')
    parser.add_argument('-beta1',           type=float,   default=0.9,              help='beta1 for ADAM')
    parser.add_argument('-beta2',           type=float,   default=0.999,            help='beta2 for ADAM')
    parser.add_argument('-weight_decay',    type=float,   default=0,                help='weight decay')
    parser.add_argument('-batch_size',      type=int,     default=4,                help='training batch size')
    parser.add_argument('-train_epoch_size',type=int,     default=1000,             help='train epoch size')
    parser.add_argument('-valid_epoch_size',type=int,     default=100,              help='valid epoch size')
    parser.add_argument('-epoch_max',       type=int,     default=100,              help='max #epochs')
    parser.add_argument('--pwcnet_path', type = str, default = './checkpoints/pwcNet-default.pytorch', help = 'the path that contains the PWCNet model')


    ### learning rate options
    parser.add_argument('-lr_init',         type=float,   default=1e-4,             help='initial learning Rate')
    parser.add_argument('-lr_offset',       type=int,     default=20,               help='epoch to start learning rate drop [-1 = no drop]')
    parser.add_argument('-lr_step',         type=int,     default=20,               help='step size (epoch) to drop learning rate')
    parser.add_argument('-lr_drop',         type=float,   default=0.5,              help='learning rate drop ratio')
    parser.add_argument('-lr_min_m',        type=float,   default=0.1,              help='minimal learning Rate multiplier (lr >= lr_init * lr_min)')
    

    ### other options
    parser.add_argument('-seed',            type=int,     default=9487,             help='random seed to use')
    parser.add_argument('-threads',         type=int,     default=8,                help='number of threads for data loader to use')
    parser.add_argument('-suffix',          type=str,     default='',               help='name suffix')
    parser.add_argument('-gpu',             type=int,     default=0,                help='gpu device id')
    parser.add_argument('-cpu',             action='store_true',                    help='use cpu?')

    opts = parser.parse_args()
    
    ### adjust options
    opts.cuda = (opts.cpu != True)
    opts.lr_min = opts.lr_init * opts.lr_min_m
    
    ### default model name
    if opts.model_name == 'none':
        
        opts.model_name = "%s_B%d_nf%d_%s" %(opts.model, opts.blocks, opts.nf, opts.norm)

        opts.model_name = "%s_T%d_%s_pw%d_%sLoss_a%s_wST%s_wHT%s_wVGG%s_L%s_%s_lr%s_off%d_step%d_drop%s_min%s_es%d_bs%d" \
                %(opts.model_name, opts.sample_frames, \
                  opts.datasets_tasks, opts.crop_size, opts.loss, str(opts.alpha), \
                  str(opts.w_ST), str(opts.w_LT), str(opts.w_VGG), opts.VGGLayers, \
                  opts.solver, str(opts.lr_init), opts.lr_offset, opts.lr_step, str(opts.lr_drop), str(opts.lr_min), \
                  opts.train_epoch_size, opts.batch_size)
    
    ### check VGG layers
    opts.VGGLayers = [int(layer) for layer in list(opts.VGGLayers)]
    opts.VGGLayers.sort()

    if opts.VGGLayers[0] < 1 or opts.VGGLayers[-1] > 4:
        raise Exception("Only support VGG Loss on Layers 1 ~ 4")

    opts.VGGLayers = [layer - 1 for layer in list(opts.VGGLayers)] ## shift index to 0 ~ 3

    if opts.suffix != "":
        opts.model_name += "_%s" %opts.suffix


    opts.size_multiplier = 2 ** 6 ## Inputs to FlowNet need to be divided by 64

    print(opts)


    torch.manual_seed(opts.seed)
    if opts.cuda:
        torch.cuda.manual_seed(opts.seed)


    ### model saving directory
    opts.model_dir = os.path.join(opts.checkpoint_dir, opts.model_name)
    print("========================================================")
    print("===> Save model to %s" %opts.model_dir)
    print("========================================================")
    if not os.path.isdir(opts.model_dir):
        os.makedirs(opts.model_dir)
    trainer.Pre_train(opts)