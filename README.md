# Learned Data Compression using MNIST dataset

This repo presents a study on learned data compression, focusing on the role of the DCT-IDCT transformation in image compression and its limitations. The aim is to develop an optimal non-linear transformation that considers complex structures and the compression scheme architecture.

### Keywords
 - Data Compression
 - DCT-IDCT Transformation
 - Variational Autoencoder

## Introduction
Image compression, a subset of data compression, is crucial for optimizing data storage and transmission. This report focuses on using and overcoming the limitations of the DCT-IDCT transformation to propose an improved approach for high-quality image compression.

## Setup

### Language and frameworks
 - TFC : TensorFlow Compression
 - Python

### MNIST Dataset
We used the MNIST handwritten digits dataset

![mnist_ex]()

# Model definition

## Before training
As we can see below, before training the randomly initialized weights are returning a noisy output insofaras the encoder/decoder has not been trained yet.

![reconstruction]()      

## Latent space dimensons
We chose a 50-dimensional latent space to represent the images which probability distribution follows Gaussian distributions.

## Encoder/Decoder architecture
The encoder performs the analysis transformation, converting images into vectors in a latent space.

### Encoder Details
 - **Convolution Layers** : Uses leaky ReLU activations, with multiple convolutional layers to encode the image into a dense latent space representation.
 - **Latent Space** : We chose a 50-dimensional latent space to represent the images which probability distribution follows Gaussian distributions.
 - **Decoder Architecture** : The decoder performs the synthesis transformation, converting vectors from the latent space back into the images space.

### Decoder Details
 - **Dense and Convolutional Layers** : Similar structure to the encoder, designed to reconstruct the image from its latent representation.
 - **Training Class** Defines a class for training with methods to compute the bitrate and distortion based on the latent representation of images.


# Compression and Reconstruction

## Noise Addition to improve robustness
Incorporates noise into the data representation to enable robust quantization and to minimize quantization errors. This kind of quantization is called **dithered quantization**. The probability distribution followed by latent representations are represented in the following figure :

 ![Noise Distribution]()

## Results after training
The model was trained on 15 epochs to get the following results :

![Results out of the decoder on the MNIST dataset]()

## Rate-Distortion trade-off
In the loss function, 'lmbda' is the parameter which tells the model how much distorsion needs to be prioretized over bits rate.

![PSNR / rate graph]()


# Generative Model Capability
Once the decoder has been trained, from a random latent vector we can generate an artificial handwritten digits with the decoder : it is a classic architecture for generative AI. We now have a model capable to generate handwritten digits

![Generated Handwritten Digits]()

# Outlook
Explores the application of variational encoders for generating handwritten digits, focusing on the initialization of the model and the role of the prior distribution in regularizing and shaping the model's complexity.

