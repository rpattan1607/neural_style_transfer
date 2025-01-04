# Neural Style Transfer
This repository contains code for transferring the style of one image to another, this is popularly known as neural style transfer. The repository also provides a streamlit app to quickt experiment with the parameters.

![image](https://github.com/user-attachments/assets/876f09fa-f615-4442-8483-47f5ef43509a)

Original Paper : https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Gatys_Image_Style_Transfer_CVPR_2016_paper.pdf

## Key features and functionalities:
* **Image preprocessing**: Handles image resizing, normalization, and other necessary preprocessing steps.
* **Image Encoding** : VGG - 19(https://paperswithcode.com/method/vgg-19) it consists of 19 layers with learnable parameters: 16 convolutional layers and 3 fully connected layers, along with 5 max-pooling layers and a softmax layer for classification. However for this task we have replaced max pooling layer with average pooling layer and only the first five convolution layers are taken.
* **Training and evaluation**: Provides scripts for training the model on a labeled dataset and evaluating its performance using metrics like content loss and style loss.
* **Inference**: Demonstrates how to use the model to transfer style for any two images.

## Requirements 

* Python
* PyTorch 
* NumPy
* PIL
* torchvision
* matplotlib
* streamlit

  ## Sample Output

  The streamlit can be run by cloning the repository and runnin the following command *streamlit run main.py*

  ![image](https://github.com/user-attachments/assets/b780fd84-bd4d-415e-afe6-4fd4a19354a4)
