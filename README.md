# Neural Style Transfer
This repository contains code for transferring the style of one image to another, this is popularly known as neural style transfer. The repository also provides a streamlit app to quickt experiment with the parameters.

## Key features and functionalities:
* Image preprocessing: Handles image resizing, normalization, and other necessary preprocessing steps.
* Image Encoding : VGG - 19 it consists of 19 layers with learnable parameters: 16 convolutional layers and 3 fully connected layers, along with 5 max-pooling layers and a softmax layer for classification. However for this task we have replaced max pooling layer with average pooling layer and only the first five convolution layers are taken.
