# Multilayer-Perceptron
By Tianyang Zhao

## Table of Contents
1. Introduction
2. Models
3. Results

## Introduction
This repository contains the Multilayer Perceptron model. The goal of the model is to decipher sign language (from 0 -> 5) .
Here are examples for each number, and how an explanation of representing the labels. These are the original pictures, before lowering the image resolutoion to 64 by 64 pixels. 


#### Note 
Note that this is a subset of the SIGNS dataset. The complete dataset contains many more signs.

### Packages Version
1. Python 3.6.9 
2. tensorflow-gpu 1.13.1
3. numpy 1.19.1
4. CUDA Version: 10.1


## Models
The model contains 4 hidden layers, and [25, 20, 14, Number_Classes] units for each layer. Using a softmax output layer, the models is able to generalizes more than two classes of outputs.

The architecture of the model is: **LINEAR -> batch_normalization -> RELU -> LINEAR -> batch_normalization -> RELU -> LINEAR -> batch_normalization -> RELU -> LINEAR -> batch_normalization -> SOFTMAX** 

By changing the parameter *layers_dims*. Users can flexible adjust the model to any depth and width as he wishes. 
The *layers_dims* used in this model is *[X_train.shape[0], 25, 20,14, Number_Classes]*. *Number_Classes = 6* as the dataset contains 6 different signs (from 0 -> 5).


With tf.train.Saver(), all the parameters and computation graph is saved in the *MLP_Softmax* folder. It saves three types of checkpoint documents automatically for each 10 iterations. When the user need to reload the trained model, he could specify the parameters of the iteration he need. Otherwise, the model will reload the parameters from the latest chckpoint.

With tf.summary(), the loss of model is recorded during the training process and save in the *Summaryfile*. With Tensorboard, the loss could be visulized easily. The user can also customize any other metrics.

You can try to put your image inside, and test your own sign image by changing the file name in line 139.

#### Note
1. GPU memory might be insufficient for extremely deep models
2. Changes of mini-batch size should impact accuracy ( minibatch_size = 32 in this model)
3. the data is randomly shuffled at the beginning of every epoch.

Keep safe and see you soon!

## Results
#### Performance on the Training set
**Train Accuracy** 0.999074

#### Performance on the Test set
**Test Accuracy**	0.716667



