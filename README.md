# MNIST, CIFAR10 Classifier using Visual Transformer 
MNIST, CIFAR10 Classifier using ViT(Visual Transformer). Please read the following paper before understanding the codebase.
[An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/abs/2010.11929)

MNIST Dataset can be found [here](https://www.kaggle.com/datasets/hojjatk/mnist-dataset)
CIFAR10 Dataset can be found [here](https://www.kaggle.com/competitions/mu-cifar10)


## Prerequisites
1. Deep understanding in Transformer Model, Attention Method 

2. Basic skill of Python

3. GPU(CUDA, CPU) with VRAM at least 300MB (You can change configuration if you don't have enough VRAM. ex. changing D_MODEL, BATCH_SIZE to smaller value)
**Important notes: Tried to run the model in Apple Metal, but had errors. Use CPU as device if using in Mac**

4. [uv-python](https://github.com/astral-sh/uv) installed in runtime(just for faster package installation)


## Data preparation
**Don't need to explicitely setup dataset.**

We are going to use MNIST or CIFAR10 Dataset from pytorch dataset library.

## Getting Started - running at Local Machine(CUDA, CPU)
1. Create venv using uv-python: 
```sh
$ uv venv 
$ source .venv/bin/activate
```

2. Install requirements
```sh
$ uv pip install -r requirements.txt
```

3. Run jupyter lab
```sh
$ jupyter lab
```

4. Setup train configuration in src/config.ipynb.

5. Go to src/Train.ipynb, and run all cells. src/Train.ipynb imports other module notebooks, so just running src/Train.ipynb is enough.

**If you want to change the parameters of training(lr, n_head, d_model etc), change it from src/config.ipynb**

## Getting Started - Google Colab
Upload colab/colab.ipynb to the runtime, and execute it!
This script is made using nbmerge with manual modification.

## Getting Started - Inference
Go to src/Inference.ipynb. Place the image to data/ directory, and set the img_path. Run all cell, then you can get the inference result of it.

## Pretrained Model - MNIST
If you want to use pretrained model for MNIST, use models/mnist_model.
(Has 94% accuracy for Test dataset)

It is trained using following configuration: 
```
DATASET="mnist"


EPOCHS = 100
BATCH_SIZE = 64
IMG_SIZE = 28
PATCH_SIZE = 4
IN_CHANNELS = 1
N_HEAD = 5
D_MODEL = 200
FFN_HIDDEN = 512 
MLP_HIDDEN = 512
N_LAYERS = 5
CLASS_NUM = 10
DROP_PROB = 0.1 
INIT_LR = 5e-5
NUM_WORKERS=2
WEIGHT_DECAY=1e-4
GRADIENT_CLIP = 1.0
```

## Pretrained Model - CIFAR10
If you want to use pretrained model for CIFAR10, use models/cifar10_model
(Has 64% accuracy for Test dataset)

It is trained using following configuration:
```
DATASET = "cifar10"

EPOCHS = 200
BATCH_SIZE = 64
IMG_SIZE = 32
PATCH_SIZE = 4
IN_CHANNELS = 3 
N_HEAD = 8
D_MODEL = 400
FFN_HIDDEN = 512 
MLP_HIDDEN = 512
N_LAYERS = 5
CLASS_NUM = 10
DROP_PROB = 0.1 
INIT_LR = 1e-4
NUM_WORKERS=2
WEIGHT_DECAY=1e-4
GRADIENT_CLIP = 1.0
```
