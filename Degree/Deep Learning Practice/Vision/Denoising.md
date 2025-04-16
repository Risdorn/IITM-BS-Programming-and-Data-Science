

# Multi-Stage Progressive Image Restoration (CVPR 2021)

## Network Architecture
<table>
  <tr>
    <td> <img src = "https://i.imgur.com/69c0pQv.png" width="500"> </td>
    <td> <img src = "https://i.imgur.com/JJAKXOi.png" width="400"> </td>
  </tr>
  <tr>
    <td><p align="center"><b>Overall Framework of MPRNet</b></p></td>
    <td><p align="center"> <b>Supervised Attention Module (SAM)</b></p></td>
  </tr>
</table>

## Installation

```
git clone https://github.com/swz30/MPRNet
cd MPRNet/Denoising
```

## Quick Run

- Download the [Datasets](Datasets/README.md)

- Download the [model](https://drive.google.com/file/d/1LODPt9kYmxwU98g96UrRA0_Eh5HYcsRw/view?usp=sharing) and place it in `./pretrained_models/`

#### Testing on SIDD dataset
- Download SIDD Validation Data and Ground Truth from [here](https://www.eecs.yorku.ca/~kamel/sidd/benchmark.php) and place them in `./Datasets/SIDD/test/`
- Run
```
python test_SIDD.py --save_images
```
#### Testing on DND dataset
- Download DND Benchmark Data from [here](https://noise.visinf.tu-darmstadt.de/downloads/) and place it in `./Datasets/DND/test/`
- Run
```
python test_DND.py --save_images
```

#### To reproduce PSNR/SSIM scores of the paper, run MATLAB script
```
evaluate_SIDD.m
```