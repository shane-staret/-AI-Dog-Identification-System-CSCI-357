# Midterm Project Group: BAGS
### Group Members: **B**ryan Birrueta, **A**ndrew Whitig, **G**o Ogata, **S**hane Staret

## Project Introduction
##### Images have long been used in the field of forensics because of the data that they are capable of capturing. Automatic digital image generation is incredibly common in the modern age and this has provided the scientific community with plenty of use cases for applying algorithms that allow information to be extracted from an image through machine learning.  
##### The implementation showcased in this project uses a neural network and deep learning (through TensorFlow and Keras) to classify images as containing dogs or not. In its current form, this project is a foundational step to building a more complex solution that can identify specific forms of wildlife within an image.

## Running the Code
##### 
=======
# Midterm Project Group: BAGS
### Group Members: **B**ryan Birrueta, **A**ndrew Whitig, **G**o Ogata, **S**hane Staret

## Project Introduction
##### Images have long been used in the field of forensics because of the data that they are capable of capturing. Automatic digital image generation is incredibly common in the modern age and this has provided the scientific community with plenty of use cases for applying algorithms that allow information to be extracted from an image through machine learning.  
##### The implementation showcased in this project uses a neural network and deep learning (through TensorFlow and Keras) to classify images as containing dogs or not. In its current form, this project is a foundational step to building a more complex solution that can identify specific forms of wildlife within an image.

## Running the Code
##### All of the actions for running the code are held in the python file TestNN.py. This file reads in the data from a directory holding the test and training images by calling methods within the ReadData.py file. The data is categorized, scaled, and separated into training and validation data during this process. The TestNN.py file then instantiates a NeuralNet object from the NeuralNet.py file and calls its methods to train and predict. As was suggested in the final presentation, adding TensorBoard can allow for the visualization of the actions performed during this process but that capability is not available in this implementation. To observe the more detail on the actions taken during the training and building of the network, change the verbose options for the model.fit and tuner.search methods from 0 to 1. At the end of the training, a graph showing the training and validation accuracies in each epoch are displayed. Finally, the accuracy on the test data is displayed in the console. The predictions of Dog or not dog are stored in the TestNN.py file as "predictions" and can be printed by uncommenting line 38 in TestNN.py. For best performance running on the Bucknell lab computers in Academic East is recommended however, the files will still run (albeit slowly) on any computer.

