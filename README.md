# Food101 Classifier using Visual Transformer 
Food101 Classifier using ViT(Visual Transformer). Please read the following paper before understanding the codebase.
[An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/abs/2010.11929)

Food101 Dataset can be found [here](https://www.kaggle.com/datasets/dansbecker/food-101)

## Prerequisites
1. Deep understanding in Transformer Model, Attention Method 

2. Basic skill of Python

3. GPU(CUDA) with VRAM 15GB (You can change configuration if you don't have enough VRAM. ex. changing D_MODEL, BATCH_SIZE to smaller value)
**Important notes: Tried to run the model in Apple Metal, but had errors. **

4. [uv-python](https://github.com/astral-sh/uv) installed in runtime(just for faster package installation)


## Data preparation
**Don't need to explicitely setup dataset.**

We are going to use Food101 Dataset from pytorch dataset library. [here](https://pytorch.org/vision/main/generated/torchvision.datasets.Food101.html)

## Getting Started - running at Local Machine(CUDA)
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

4. Go to src/Train.ipynb, and run all cells. src/Train.ipynb imports other module notebooks, so just running src/Train.ipynb is enough.

**If you want to change the parameters of training(lr, n_head, d_model etc), change it from src/config.ipynb**

## Getting Started - Google Colab
Upload colab/colab.ipynb to the runtime, and execute it!
This script is made using nbmerge with manual modification.

## Getting Started - Inference
Go to src/Inference.ipynb. Place the image to data/ directory, and set the img_path. Run all cell, then you can get the inference result of it.
