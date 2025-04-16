# LaKDNet: Revisiting Image Deblurring with an Efficient ConvNet

 
### Prerequisites

![Ubuntu](https://img.shields.io/badge/Ubuntu-16.0.4%20&%2018.0.4-blue.svg?style=plastic)
![Python](https://img.shields.io/badge/Python-3.8.13-yellowgreen.svg?style=plastic)
![CUDA](https://img.shields.io/badge/CUDA-11.1.1%20-yellowgreen.svg?style=plastic)
![PyTorch](https://img.shields.io/badge/PyTorch-1.8.0-yellowgreen.svg?style=plastic)

Notes: the code may also work with other library versions that didn't specify here.

#### 1. Installation

Clone this project to your local machine

```bash
$ git clone https://github.com/lingyanruan/LaKDNet.git
$ cd LaKDNet
```
#### 2. Environment setup

```bash
$ conda create -y --name LaKDNet python=3.8.13 && conda activate LaKDNet
$ sh install_CUDA11.1.1.sh
# Other version will be checked and updated later.
```


#### 3. Pre-trained models

Download and unzip under `./ckpts/` from [webpage](https://lakdnet.mpi-inf.mpg.de/):

#### 4. Datasets download

Download and unzip under `./Test_sets/` from [webpage](https://lakdnet.mpi-inf.mpg.de/):

The original full defocus datasets could be found here: ([LFDOF](https://sweb.cityu.edu.hk/miullam/AIFNET/), [DPDD](https://github.com/Abdullah-Abuolaim/defocus-deblurring-dual-pixel), [CUHK](http://www.cse.cuhk.edu.hk/~leojia/projects/dblurdetect/dataset.html) and [RealDOF](https://www.dropbox.com/s/arox1aixvg67fw5/RealDOF.zip?dl=1)):


The original full motion datasets could be found here: ([GOPRO](https://seungjunnah.github.io/Datasets/gopro), [HIDE](https://github.com/joanshen0508/HA_deblur?tab=readme-ov-file), [REALR&REALJ](http://cg.postech.ac.kr/research/realblur/)).
#### 5. Command Line

```bash
# type could be Motion or Defocus 
# for motion evaluation
$ python run.py --type Motion

# for defocus evaluation
$ python run.py --type Defocus

```



