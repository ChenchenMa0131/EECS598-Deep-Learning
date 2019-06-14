# Generating Chinese Ancient Ink Painting from Sketch Using Conditional GANs

A course project of W19 EECS598-012 Deep Learning of Team 03

The basic framework is adapted from [Jun-yan Zhu]'s (https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix). We changed the original repo by following: added various loss metrics to pix2pix model, developed new models Ske2Ink and Ske2InkRandom, and created a new dataloader.

## Training and Testing commands

### To train the model with existing datasets(for example datasets/flower_paint), use the following command:

```
$ python train.py --dataroot ./datasets/flower_paint/ --name pix2pix_L1 --model pix2pix --direction BtoA --display_id 0 --gpu_ids -1
```

--dataroot: the path for datasets

--name: the name of the trained model

--model: specify the model

--direction: specify the training direction, BtoA means transforming B to A, here A is ground truth, B is sketch image

--display_id: for disabling visdom

--gpu_ids: -1 for using CPU, delete this if using GPU

### To test the model, use the following command

```
$ python test.py --dataroot ./datasets/flower_paint/ --name pix2pix_L1 --model pix2pix --direction BtoA
```

## Models and Corresponding datasets

### Original baseline and applying additional losses on pix2pix model and flower_paint dataset

Our baseline uses ./datasets/flower_paint dataset which contains only flower images. The following command shows training a model with L1 loss weight 0 and L2 loss weight 50

```
$ python train.py --dataroot ./datasets/flower_paint/ --name pix2pix_L2 --model pix2pix --direction BtoA --display_id 0 --lambda_L1 0 --lambda_L2 50
```

We explored following loss metrics

#### Loss metrics
--lambda_L1: L1 loss weight, default 100

--lambda_L2: L2 loss weight, default 0

--lambda_SL1: SL1 loss weight, default 0

--lambda_style: Style loss weight, default 0

### Image to image translation with category embeddings
We developed a new model in Ske2Ink_model.py for mixed category dataset ./datasets/flower_mountain, which included a category embedding in the image embedding vector. The training command is shown below:

```
$ python train.py --dataroot ./datasets/flower_mountain/ --name Ske2Ink --model Ske2Ink --direction BtoA --display_id 0 --lambda_category 10
```

#### Loss metrics
--lambda_category: weight for category loss

### Diverse Image to image translation with category embeddings
We modified the previous model by adding an random latent vector to the image embedding vector in Ske2InkRandom_model.py for mixed category dataset ./datasets/flower_mountain. The training command is shown below:

```
$ python train.py --dataroot ./datasets/flower_mountain/ --name Ske2Ink_Random --model Ske2InkRandom --direction BtoA --display_id 0 --lambda_category 10 --lambda_random 1
```

#### Loss metrics
--lambda_category: weight for category loss

--lambda_random: weight for random vector loss
