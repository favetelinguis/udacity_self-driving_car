# **Traffic Sign Recognition** 

## Writeup

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./labeldist.png "Label Dist"
[image2]: ./examples/grayscale.jpg "Grayscaling"
[image3]: ./examples/random_noise.jpg "Random Noise"
[image4]: ./signs/3speedlimit60.png "Speedlimit 60"
[image5]: ./signs/9nopassing.png "No Passing"
[image6]: ./signs/18caution.png "Caution"
[image7]: ./signs/25roadwork.png "Roadwork"
[image8]: ./signs/34turnleftahead.png "Turn Left Ahead"
[image9]: ./signs/41endofnopassing.png "End Of No passing"
[image10]: ./heatmaps/24heatmapImage.jpg "Wrong Bumpy Road"
[image11]: ./heatmaps/25heatmapImage.jpg "Right Road Work"
[image12]: ./predictions.png "Predictions"
[image13]: ./inneractivations/l1.png "L1"
[image14]: ./inneractivations/l2.png "L2"
[image15]: ./inneractivations/l3.png "L3"

---

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the numpy library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799 
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

The below picture shows an exploratory visualization of the data set. It is a bar chart showing the distribution of lables between train, validation and tests set. As shown in the graphs the label distribution looks to be the same between datasets.

![alt text][image1]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

Rescaling is done by multiplying all channels with 1/255 to get numbers in the range 0-1 since
neural networks prefere working with small numbers.
Normalization is done automatically with the flow function from ImageDataGenerator.

#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution 3x3     	| 1x1 stride, valid padding, outputs 30x30x32 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 15x15x32 				|
| Convolution 2x2     	| 1x1 stride, valid padding, outputs 14x14x128 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 7x7x128    				|
| Convolution 2x2     	| 1x1 stride, valid padding, outputs 6x6x128 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 3x3x128    				|
| Flatten   	      	| outputs 1152    				                |
| Dropout   	      	| outputs 1152    				                |
| Fully connected		| outputs 256  									|
| RELU					|												|
| Fully connected		| outputs 128  									|
| RELU					|												|
| Fully connected		| outputs 64   									|
| RELU					|												|
| Softmax				| outputs 43   									|
|						|												|
|						|												|
 

#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

While training I used the Adam optimizer with a categorical crossentropy loss function and a learning rate of 0.001. Since im running on my local computer wich has a GTX1070 I wanted try a bigger batch sice since I know I have enogh memory on my card and using a bigger batch size gives me more stable updates. I did run for 50 epoch. I reached an accuracy of 90% after only 13 epoch but to get passt 93% validation accuracy I needed to continue for 50 epochs.

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were (can be found in cell 10):
* training set accuracy of 99.36%
* validation set accuracy of 94.97%
* test set accuracy of 91.00%

I did not select a well known architecure since I just wanted to play around with different architectures and se what effects they had on the final result. However it turned out that mostly everything CNN manage to get an accuracy of 90% but getting past 93% was much harder. Below I will summarize the evolution of my solution.
The first thing I tried was using 2 64 channels Convolution layers with maxpooling followed by 2 fully connected layers with 128 respective 64 cells. This worked decent but I was not able to push up the accuracy to 93% withour overfitting to the training data. I decided to make the network bigger and add regularization. The reason I could not just make the network bigger is since then I would overfit even worse so I needed regularization. I added a dropout layer after the flattening layer and now I could train my network much longer without facing overfitting. The next step I did was trying to add many different data augmentations wich is simple with the ImageDataGenerator. However I newer manage to get any good result with this. Often I got stuck at an accuracy of 80% and if I did get over 93% it was because the augmentations I generated where not that different from the training set and gave no extra generalization on the test set.
The parameters I tuned the most where number of channels in the different CNN layers seeing if the network benefitted from having more feature maps, but I guess the training images are pretty simple and I did not need an enormus network. I did also expeiment with the number of cells in the dense layers but there where really no need to go bigger there either since I found no impact on the validation accuracy.
I think CNN are a very good choice for this task and the reason I selected to use CNN is beacuse they are indifferent to where information is located in the image. That is they dont learn where in the image some shape appear only that that shape appear somewhere. If I would have use only fully connected layers the same sign at different locations in the image would need to be learned.

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are six German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] ![alt text][image9] 
![alt text][image7] ![alt text][image8]

The first four images where choosen to see how well the classifier would work when there is no extra noice in the image. For the training data there was often other things in the picure except the actual trafic sign. Im interested to see if the classifier has been able to understand that it is only the actual sign that is of interest. Im not expecting these images to be hard to classify and they work as a baseline of what I expect my network to be able to predict. 
The fifth image might be hard to get right since it has pices of several other signs in the picture.
The sixth image might be hard since it is a blue sign on a blue background.

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Speedlimit 60    		| Speedlimit 30									| 
| No Passing   			| No Passing									|
| Yield					| Caution										|
| Roadwork       		| Roadwork					 			    	| 
| Turn Left Ahead		| Turn Left Ahead	        					|
| End Of No passing		| End Of No passing	        					|


The model was able to correctly guess 5 of the 6 traffic signs, which gives an accuracy of 83%. This compares favorably to the accuracy on the test set of 91%.

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 12th cell of the Ipython notebook. The explanation on the y-axis is the actual label.

![alt text][image12]

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?
Looking at the images below the edges of the sign really stands out wich I find interesting. There are many signs with this shape. What I would have thought where that the man in the middle holding the shovel should have standed out more.

![alt text][image13]
![alt text][image14]
![alt text][image15]

### (Extra for fun) [Grad-CAM](https://arxiv.org/abs/1610.02391) debugging
#### To get more insight into how the network made wrong predictions I implemented the Grad-CAM method.
I created a second model excluding the dropout layer to play around with Grad-CAM.
This second model called bad_model in the code predicted Bumpy Road when the actual prediction should have been Roadword. Lets see what aspects of the actual image lead the model to belive it was a sign of a bumpy road.

![alt text][image10]

I find it strange that the feature in the image that is most telling of bumpy road is the area arond the shovel.
Maybe it is that the model is not taking into account that there is also a person on the sign?

As a the second guess the model predicted the right image. The below image shows the aspects of the image
that was used by the model to come to this conclution. Here the center of the sign is not of that much interest. Instead the corners of the sign are more interesting. 

I think that to truly understand and be able to read the heatmaps produced by Grad-CAM takes some practice.

![alt text][image11]

