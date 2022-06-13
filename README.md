# Learning Blind Video Temporal Consistency with PWCNet


### Table of Contents
1. [Introduction](#introduction)
1. [Requirements and Dependencies](#requirements-and-dependencies)
1. [Installation](#installation)
1. [Dataset](#dataset)
1. [Apply Pre-trained Models](#apply-pre-trained-models)
1. [Training and Testing](#training-and-testing)
1. [Evaluation](#evaluation)
1. [Image Processing Algorithms](#image-processing-algorithms)
1. [Acknowledge](#acknowledge)


### Introduction
Our method is written based on [Learning Blind Video Temporal Consistency](https://github.com/phoenix104104/fast_blind_video_consistency), which takes the original unprocessed and per-frame processed videos as inputs to produce a temporally consistent video. Our approach is agnostic to specific image processing algorithms applied on the original video. The difference from our method to LBVC is that our method uses PWC-Net, a newer method to calculate the optical flow on pytorch1.0.0 and python3.6.


### Requirements and dependencies
- [Pytorch 1.0.0](https://pytorch.org/)
- [TensorboardX](https://github.com/lanpa/tensorboardX)
- [pytorch-pwc](https://github.com/sniklaus/pytorch-pwc) (Code for [PWC-Net](https://github.com/NVlabs/PWC-Net) on pytorch1.0 and python2.0)
- [LPIPS](https://github.com/richzhang/PerceptualSimilarity) (for evaluation)

Our code is tested on Ubuntu 16.04 with cuda 9.0 and cudnn 7.0.


### Installation
Download repository:

    git clone git@github.com:zhaoyuzhi/Learning-Blind-Video-Temporal-Consistency-with-PWCNet.git

### Dataset

    cd data
    ./download_data.sh [train | test | all]
    cd ..

I used the same dataset as LBVC did. The dataset include train and test. The Dataset including origin frames and per-frame processed videos.
    
### Apply pre-trained models
I have trained the W3_D1_C1_I1 task, and the trained model are in:
https://drive.google.com/file/d/1Nz6vqzIR0p2vGWIiQDwc6pV-i5DsOAw_/view?usp=sharing

You can just download it in the folder of this repository and unzip it to get the pre-trained model.

Test pre-trained model:

    python test.py -method trained_model_example -epoch 100 -dataset DAVIS -task colorization/ECCV16
    
The output frames are saved in `data/test/trained_model_example/epoch_100/colorization/ECCV16`.


### Training and testing
Train a new model:

    python train.py -datasets_tasks W3_D1_C1_I1

The default parameters are specified in train.py. `lists/train_tasks_W3_D1_C1_I1.txt` specifies the dataset-task pairs for training.

Test a model:

    python test.py -method MODEL_NAME -epoch N -dataset DAVIS -task WCT/wave
    
Check the checkpoint folder for the `MODEL_NAME`.
The output frames are saved in `data/test/MODEL_NAME/epoch_N/WCT/wave/DAVIS`.


You can also generate results for multiple tasks using the following script:

    python batch_test.py -method output/MODEL_NAME/epoch_N

which will test all the tasks in `lists/test_tasks.txt`.


### Evaluation
**Temporal Warping Error**

To compute the temporal warping error, we first need to generate optical flow and occlusion masks:

    python compute_flow_occlusion.py -dataset DAVIS -phase test

The flow will be stored in `data/test/fw_flow/DAVIS`. The occlusion masks will be stored in `data/test/fw_occlusion/DAVIS`.

Then, run the evaluation script:

    python evaluate_WarpError.py -method output/MODEL_NAME/epoch_N -task WCT/wave
    
**LPIPS**

Download [LPIPS repository](https://github.com/richzhang/PerceptualSimilarity) and change `LPIPS_dir` in evalate_LPIPS.py if necesary (default path is `../LPIPS`).

Run the evaluation script:

    python evaluate_LPIPS.py -method output/MODEL_NAME/epoch_N -task WCT/wave

**Batch evaluation**

You can evaluate multiple tasks using the following script:

    python batch_evaluate.py -method output/MODEL_NAME/epoch_N -metric LPIPS
    python batch_evaluate.py -method output/MODEL_NAME/epoch_N -metric WarpError
    
which will evaluate all the tasks in `lists/test_tasks.txt`.

   
### Test on new videos
To test our model on new videos or applications, please follow the folder structure in `./data`.

Given a video, we extract frames named as `%05d.jpg` and save frames in `data/test/input/DATASET/VIDEO`.

The per-frame processed video is stored in `data/test/processed/TASK/DATASET/VIDEO`, where `TASK` is the image processing algorithm applied on the original video.


### Image Processing Algorithms
We use the following algorithms to obtain per-frame processed results:

**Style transfer**
- [WCT: Universal Style Transfer via Feature Transforms, NIPS 2017](https://github.com/Yijunmaverick/UniversalStyleTransfer)
- [Fast Neural Style Transfer: Perceptual Losses for Real-Time Style Transfer and Super-Resolution, ECCV 2016](https://github.com/jcjohnson/fast-neural-style)

**Image Enhancement**
- [DBL: Deep Bilateral Learning for Real-Time Image Enhancement, Siggraph 2017](https://groups.csail.mit.edu/graphics/hdrnet/)

**Intrinsic Image Decomposition**
- [Intrinsic Images in the Wild, Siggraph 2014](http://opensurfaces.cs.cornell.edu/publications/intrinsic/)

**Image-to-Image Translation**
- [CycleGAN: Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks, ICCV 2017](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix)

**Colorization**
- [Colorful Image Colorization, ECCV 2016](https://github.com/richzhang/colorization)
- [Let there be Color!: Joint End-to-end Learning of Global and Local Image Priors for Automatic Image Colorization with Simultaneous Classification, Siggraph 2016](http://hi.cs.waseda.ac.jp/~iizuka/projects/colorization/en/)

### Acknowledge
Thank for the team of [LBVC](https://github.com/phoenix104104/fast_blind_video_consistency), their test give a new idea to deal the problem of video temporal consistency. And thank for The team of [PWC-Net](https://github.com/NVlabs/PWC-Net) providing a more efficient method to calculate optical flow.

    
