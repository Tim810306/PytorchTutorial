<p align="center"><img width="40%" src="logo/pytorch_logo.png" /></p>

--------------------------------------------------------------------------------

This repository provides tutorial code for deep learning researchers to learn [PyTorch](https://github.com/pytorch/pytorch). In the tutorial, most of the models were implemented with less than 30 lines of code. Before starting this tutorial, it is recommended to finish [Official Pytorch Tutorial](http://pytorch.org/tutorials/beginner/deep_learning_60min_blitz.html).


<br/>

## Table of Contents
#### PPTs
* [PyTorchZeroToAll](https://drive.google.com/drive/folders/0B41Zbb4c8HVyUndGdGdJSXd5d3M)
#### Videos
* [PyTorchZeroToAll-lecture1~lecture4-Chinese version](https://youtu.be/MuyUFqJf_Ug)
#### Install
* [PyTorch install]
## Getting Installed for OSX pip 3.6 without coda
```bash
$ pip3 install http://download.pytorch.org/whl/torch-0.3.1-cp36-cp36m-macosx_10_7_x86_64.whl 
$ pip3 install torchvision 
```
## Getting Installed for Linux pip 3.5 with coda9
```bash
$ pip3 install http://download.pytorch.org/whl/cu90/torch-0.3.1-cp35-cp35m-linux_x86_64.whl
$ pip3 install torchvision 
```
#### 1. Basics
* [Lecture2:Linear Model](https://github.com/Tim810306/PytorchTutorial/tree/master/tutorials/01-basics/Linear_Model/main.py)
* [Lecture3:Gradient_Descent](https://github.com/Tim810306/PytorchTutorial/tree/master/tutorials/01-basics/Gradient_Descent/main.py)
* [Lecture4:Back_propagration](https://github.com/Tim810306/PytorchTutorial/tree/master/tutorials/01-basics/Back_propagration/main.py)
* [Lecture5:Linear_regression](https://github.com/Tim810306/PytorchTutorial/tree/master/tutorials/01-basics/Linear_regression/main.py)
* [Lecture6:Logistic_regression](https://github.com/Tim810306/PytorchTutorial/tree/master/tutorials/01-basics/Logistic_regression/main.py)
* [Lecture7:Wide_Deep](https://github.com/Tim810306/PytorchTutorial/tree/master/tutorials/01-basics/Wide_Deep/main.py)
* [Lecture8:DataLoader](https://github.com/Tim810306/PytorchTutorial/tree/master/tutorials/01-basics/DataLoader/main.py)
* [Lecture8:DataLoader_logistic](https://github.com/Tim810306/PytorchTutorial/tree/master/tutorials/01-basics/DataLoader/main_logistic.py)
* [Lecture9:Softmax_Classifier](https://github.com/Tim810306/PytorchTutorial/tree/master/tutorials/01-basics/Softmax_Classifier/main.py)
* [Lecture9:Softmax_Classifier_mnist](https://github.com/Tim810306/PytorchTutorial/tree/master/tutorials/01-basics/Softmax_Classifier/main_mnist.py)

#### 2. Intermediate
* [Convolutional Neural Network](https://github.com/Tim810306/PytorchTutorial/tree/master/tutorials/02-intermediate/convolutional_neural_network/main.py#L33-L53)
* [Deep Residual Network](https://github.com/Tim810306/PytorchTutorial/tree/master/tutorials/02-intermediate/deep_residual_network/main.py#L67-L103)
* [Recurrent Neural Network](https://github.com/Tim810306/PytorchTutorial/tree/master/tutorials/02-intermediate/recurrent_neural_network/main.py#L38-L56)
* [Bidirectional Recurrent Neural Network](https://github.com/Tim810306/PytorchTutorial/tree/master/tutorials/02-intermediate/bidirectional_recurrent_neural_network/main.py#L38-L57)
* [Language Model (RNN-LM)](https://github.com/Tim810306/PytorchTutorial/tree/master/tutorials/02-intermediate/language_model/main.py#L28-L53)
* [Generative Adversarial Network](https://github.com/Tim810306/PytorchTutorial/blob/master/tutorials/02-intermediate/generative_adversarial_network/main.py#L34-L50)

#### 3. Advanced
* [Image Captioning (CNN-RNN)](https://github.com/Tim810306/PytorchTutorial/tree/master/tutorials/03-advanced/image_captioning)
* [Deep Convolutional GAN (DCGAN)](https://github.com/Tim810306/PytorchTutorial/tree/master/tutorials/03-advanced/deep_convolutional_gan)
* [Variational Auto-Encoder](https://github.com/Tim810306/PytorchTutorial/tree/master/tutorials/03-advanced/variational_auto_encoder)
* [Neural Style Transfer](https://github.com/Tim810306/PytorchTutorial/tree/master/tutorials/03-advanced/neural_style_transfer)

#### 4. Utilities
* [TensorBoard in PyTorch](https://github.com/Tim810306/PytorchTutorial/tree/master/tutorials/04-utils/tensorboard)


<br/>

## Getting Started
```bash
$ git clone https://github.com/Tim810306/PytorchTutorial.git
$ cd PytorchTutorial/tutorials/project_path
$ python main.py               # cpu version
$ python main-gpu.py           # gpu version
$ python main_XXX.py           # execute XXX for cpu version
```

<br/>

## Dependencies
* [Python 2.7 or 3.5](https://www.continuum.io/downloads)
* [PyTorch 0.1.12](http://pytorch.org/)



<br/>


## Author
Cheng Yu Ting/ [@Tim810306](https://github.com/Tim810306)