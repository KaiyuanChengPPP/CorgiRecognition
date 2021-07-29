# CorgiRecognition
It is a project using convulotional neural network to recognize corgi dogs' species
In this assignment I used pytorch to help me implement the network, and Pillow to help me read images. I used cpu instead of cuda to help me process the data, since I have an AMD RADEON Graphics. 
There are two convolutional layers in my neural network, the first one has 3 in_channels (RDB), 12 out_channels and features. The second layer has 12 in_channels, 24 features and out_channels. There are no pooling layers implemented. 
I used Variable to get the gradient, ADAM for optimizer, cross entropy for loss function. The model which has the highest accuracy of prediction will be kept, and saved as a model, which will then be used to predict the pictures. The accuracy of prediction is not stable, since I only did 10 epochs for saving times. It varies from 85% to 100%. 
