# Learning Blind Video Temporal Consistency with PWCNet


### Requirements and dependencies
- [Pytorch 1.0.0](https://pytorch.org/)
- [TensorboardX](https://github.com/lanpa/tensorboardX)
- [pytorch-pwc](https://github.com/sniklaus/pytorch-pwc) (Code for [PWC-Net](https://github.com/NVlabs/PWC-Net) on pytorch1.0 and python2.0)
- [LPIPS](https://github.com/richzhang/PerceptualSimilarity) (for evaluation)

Our code is tested on Ubuntu 16.04 with cuda 9.0 and cudnn 7.0.


## Test

To compare with other methods, please follow:

1. create new folders: **input / processed / output**
```bash
mkdir data
cd data
mkdir input
mkdir processed
mkdir output
```

2. take colorization as an example:

- **input** folder contain `grayscale` frames
```bash
put input grayscale folders in `./data/input` as:
input/DAVIS
input/videvo
```

- **processed** folder contain `colorized` frames (by specific methods, e.g., CIC)
```bash
put generated frames folders in `./data/processed` as:
processed/DAVIS
processed/videvo
```

- **output** folder contain refined frames by LBVC-Sheroa (after testing it will contain output images)

- run:
```bash
sh test1.sh
sh test2.sh
```

3. download results in **output** folder, e.g., the results should be CIC+BTC

