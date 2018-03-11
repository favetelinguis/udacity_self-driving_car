# **Behavioral Cloning** 

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./modelarchitecture.png "Model Architecture"
[image2]: ./pic0_steering_0.0.jpg "Center camera"
[image3]: ./center_1.jpg "Recovery Image"
[image4]: ./center_2.jpg "Recovery Image"
[image5]: ./center_3.jpg "Recovery Image"
[image8]: ./validationloss10runs.png "Validation loss"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* README.md summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model is based on the [Nvida](https://devblogs.nvidia.com/deep-learning-self-driving-cars/) architecture (line 48-67). It starts with a lambda layer for normalizing input images and then a Cropping layer to remove the top most part of incomming images. This is done since the sky does probably not contain meaning ful information about the how to predict steering anges. Following are 5 convolution layers for feature extraction. The first 3 layers uses a 2x2 stride and a 5x5 kernel and the last to uses no stride with a 3x3 kernel. Next are 4 dense layers followed by a linear single neuron dense layer for the regression. All convolutional layers and all but the last dense layer uses relu activation functions to intruduce non-linearity in the model.

#### 2. Attempts to reduce overfitting in the model

No dropout or pooling layers are added since I wanted to stay true to the original Nvidia architecture. Early stopping and more training data was the only "regularization" I used to prevent overfitting.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 79).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road and I used the images from all 3 cameras.

For details about how I created the training data, see the next section.

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to build on what large players such as Nvidia and comma.ai has already built.

My first step was to use a convolution neural network model similar to the [Nvida](https://devblogs.nvidia.com/deep-learning-self-driving-cars/) I thought this model might be appropriate because it has been proven to be deep enogh to encode real life driving.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. The most common problem I had was that the model worked find on both the training set and the validation set but failed to generalize to actually driving in the simulator, how I solved this will be discussed in point 3 below.

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track. The most common issue was with the turns where the car ventured outside the road and then faild to find its way back. To improve the driving behavior in these cases, I added more training data where I recovered back to the road after steering out from the road.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

Below are two videos showing the early model vs the end model running in the simulator:

[Failure!!!](https://www.youtube.com/watch?v=HETa9p4GmZc)

[Success!!!](https://www.youtube.com/watch?v=aiUu7oWepxE&t=1s)

#### 2. Final Model Architecture

The final model architecture consisted of a convolution neural network with the following layers and layer sizes ...
| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 160x320x3 RGB image   						| 
| Lambda Normalization	| 160x320x3               						| 
| Cropping 2D       	| 65x320x3               						| 
| Convolution 5x5     	| 2x2 stride, valid padding, outputs 31x158x24 	|
| RELU					|												|
| Convolution 5x5     	| 2x2 stride, valid padding, outputs 14x77x36 	|
| RELU					|												|
| Convolution 5x5     	| 2x2 stride, valid padding, outputs 5x37x58 	|
| RELU					|												|
| Convolution 3x3     	| outputs 3x35x64                           	|
| RELU					|												|
| Convolution 3x3     	| outputs 1x33x64                           	|
| RELU					|												|
| Flatten   	      	| outputs 2112    				                |
| Fully connected		| outputs 1164 									|
| RELU					|												|
| Fully connected		| outputs 100  									|
| RELU					|												|
| Fully connected		| outputs 50   									|
| RELU					|												|
| Fully connected		| outputs 10   									|
| RELU					|												|
| Fully connected		| outputs 1   									|
|						|												|
|						|												|

Here is a visualization of the architecture.

![alt text][image1]

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded one lap on track one using center lane driving, then I did the same thing driving in the opposite direction. Here is an example image of center lane driving:

![alt text][image2]

For the training I also used the images from the left and right cameras where I used an adjutment factor of 0.2.
That is for the left image I added 0.2 to the center streering angle and for the right camera I subtracted 0.2.

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to recover from when things go wrong. This is important to avoid distribution drift in the training data. For example if the training data only contains pictures of the car driving on the road the model will fail as soon as it makes a mistake and drifts of the road. These images show what a recovery looks like starting from far of to the left side:

![alt text][image3]
![alt text][image4]
![alt text][image5]

After the collection process, I had 9177 number of data points. I then preprocessed this data by cropping the top part of the image and normalizing the data.

I finally randomly shuffled the data set and put 20% of the data into a validation set.

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 5 as evidenced by the validation loss show below I used an adam optimizer so that manually adjusting the learning rate wasn't necessary.

![Validation loss over 10 epochs][image8]