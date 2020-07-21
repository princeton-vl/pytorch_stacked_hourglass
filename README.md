# Stacked Hourglass Networks in Pytorch

Based on **Stacked Hourglass Networks for Human Pose Estimation.** [Alejandro Newell](http://www-personal.umich.edu/~alnewell/), Kaiyu Yang, and [Jia Deng](https://www.cs.princeton.edu/~jiadeng/). *European Conference on Computer Vision (ECCV)*, 2016. [Github](https://github.com/princeton-vl/pose-hg-train)

PyTorch code also adopted from: **Associative Embedding: End-to-end Learning for Joint Detection and Grouping.**
[Alejandro Newell](http://www-personal.umich.edu/~alnewell/), Zhiao Huang, and [Jia Deng](https://www.cs.princeton.edu/~jiadeng/). *Neural Information Processing Systems (NeurIPS)*, 2017. [Github](https://github.com/princeton-vl/pose-ae-train)

## Getting Started

This repository provides everything necessary to train and evaluate a single-person pose estimation model on MPII. If you plan on training your own model from scratch, we highly recommend using multiple GPUs.

Requirements:

- Python 3 (code has been tested on Python 3.8.2)
- PyTorch (code tested with 1.5)
- CUDA and cuDNN (tested with Cuda 10)
- Python packages (not exhaustive): opencv-python (tested with 4.2), tqdm, cffi, h5py, scipy (tested with 1.4.1), pytz, imageio

Structure:
- ```data/```: data loading and data augmentation code
- ```models/```: network architecture definitions
- ```task/```: task-specific functions and training configuration
- ```utils/```: image processing code and miscellaneous helper functions
- ```train.py```: code for model training
- ```test.py```: code for model evaluation

#### Dataset
Download the full [MPII Human Pose dataset](http://human-pose.mpi-inf.mpg.de/), and place the images directory in data/MPII/

#### Training and Testing

To train a network, call:

```python train.py -e test_run_001``` (```-e,--exp``` allows you to specify an experiment name)

To continue an experiment where it left off, you can call:

```python train.py -c test_run_001```

All training hyperparameters are defined in ```task/pose.py```, and you can modify ```__config__``` to test different options. It is likely you will have to change the batchsize to accommodate the number of GPUs you have available.

Once a model has been trained, you can evaluate it with:

```python test.py -c test_run_001```

The option "-m n" will automatically stop training after n total iterations (if continuing, would look at total iterations)

#### Pretrained Models

An 8HG pretrained model is available [here](http://www-personal.umich.edu/~cnris/original_8hg/checkpoint.pt). It should yield validation accuracy of 0.902.

A 2HG pretrained model is available [here](http://www-personal.umich.edu/~cnris/original_2hg/checkpoint.pt). It should yield validation accuracy of 0.883.

Models should be formatted as exp/<exp_name>/checkpoint.pt

Note models were trained using batch size of 16 along with Adam optimizer with LR of 1e-3 (instead of RMSProp at 2.5e-4), as they outperformed in validation. Code can easily be modified to use original paper settings. The original paper reported validation accuracy of 0.881, which this code approximately replicated. Above results also were trained for approximately 200k iters, while the original paper trained for less.

#### Training/Validation split

The train/val split is same as that found in authors' [implementation](https://github.com/princeton-vl/pose-hg-train)

#### Note

During training, occasionaly "ConnectionResetError" warning was occasionally displayed between epochs, but did not affect training.  
