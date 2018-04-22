
---

**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./car.png
[image11]: ./notcar.png
[image2]: ./hlshog.png
[image3]: ./slidingwindow.png
[image4]: ./example.png
[image5]: ./heatmap.png
[image6]: ./labels.png
[image7]: ./labelboxes.png
[video1]: ./project_video.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  

You're reading it!

### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contained in function `get_hog_features` in the file called `lesson_functions.py`.  

I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][image1]
![alt text][image11]

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  

Here is an example using the `HLS` color space and HOG parameters of `orientations=9`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:

![alt text][image2]

#### 2. Explain how you settled on your final choice of HOG parameters.

After visual inspection I found an orientation of 9 and 8 px per cell looks to be consistently strong for light and dark examples, 2 block looks to be standard so from online guides so I will stick with this. As for color space channels using carrying light informations is struggling so I will use all channels. YUV and YCrCb worked good on images but did not generalize to video and I found HLS to work better on the video.

#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I trained a linear SVM in the file `classifier.py` in the function `training`. I used the following configuration of features.
```python
color_space = 'HLS' # Color space
orient = 9          # HOG orientations
pix_per_cell = 8    # HOG pixels per cell
cell_per_block = 2  # HOG cells per block
hog_channel = 'ALL' # What channel to draw HOG features
spatial_size = (16, 16) # Spatial binning dimensions
hist_bins = 32           # Number of histogram bins
spatial_feat = True      # Spatial features on or off
hist_feat = True         # Histogram features on or off
hog_feat = True          # HOG features on or off
```
Using this configuration I reached a test Accuracy of 98.5%

I did experiment with different values of C using grid search but no noticiable difference could be found trying different values of C as can be seen in the function `train_with_grid_search` in `classifier.py`.

### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

The code is located in cell 16 in `VehicleDetection.ipynb`.
This step was very much a trial and error step where the trade off was how many false negative I got. If I used anything less then a scale of 1.2 the number of false negative was to high. The same thing is true with the overlap, I just got to many false nagatives on my classifier even thogh the result on the test set was very good so I was trying to use as little overlap as possible to isolate only the cars.

![alt text][image3]

#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on two scales using HLS 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result. As for performace I tried using less features but that just mabe my classifier worse. It would be interesting to apply for example PCA to narrow down the number of features used. Here is an example image:

![alt text][image4]
---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./output_video/project_video.mp4)


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  
For the video I used a `deque` of 20 frames to store all bouding boxes found possetive over the last 20 frames. Then for each new frame I apply to `heat_pipline` function over all frames.

This code can be found in `heat_pipeline` in `lesson_functions.py`.

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

### Here is one frames and the corresponding heatmap:

![alt text][image4]
![alt text][image5]

### Here is the output of `scipy.ndimage.measurements.label()` on the integrated heatmap:
![alt text][image6]

### Here the resulting bounding boxes are drawn onto the frame:
![alt text][image7]



---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

I had so much problems with false posetives. The classifier worked fine for the test set but once I applied it on unseen images it just performed horrible. Even if aggregated multiple framed and increased the threshold lots I still got many false posetive. The places my pipeline will fale is on rails on the road but also on lane line and meeting trafic. It would be very interesting to see how a deep learning approach to this problem would work after having tried doing it the old way of manually extracting features and training a shallow model.