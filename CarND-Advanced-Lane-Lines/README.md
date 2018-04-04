**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./output_images/calibration1.png "Undistorted"
[image2]: ./output_images/test1.png "Road Transformed"
[image3]: ./threshold.png "Binary Example"
[image4]: ./warped.png "Warp Example"
[image5]: ./polinomial.png "Fit Visual"
[image6]: ./linesdrawn.png "Output"
[video1]: ./project_video.mp4 "Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.

You're reading it!

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained in the third code cell of the IPython notebook located in "./main.ipynb".  

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners found with `findChessboardCorners` in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result: 

![alt text][image1]

### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

To demonstrate this step I demonstrate a distortion corrected image below.
![alt text][image2]

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

I used a combination of color and gradient thresholds to generate a binary image (thresholding steps at cells 6 through 12 in `main.ipynb`).  Here's an example of my output for this step.

![alt text][image3]

I started with looking at different color spaces. The ones I testad where RGB, HLS, HSV and LAB. After a visual inspection of the different channels in each color space I found the R channel in RGB to be a good fit and the s channel in HLS. After some experimenting I used thresholds of `(220, 255)` resp. `(100, 255)` .
For gradient thresholds I used the Sobel operation in both x and y direction the magnitude and the direction of the gradients with the following thresholds `(25, 100)`, `(50, 150)`, `(50, 250)` and `(0.5, 1.3)`. I found the directional gradient to be extremely noicy and even thogh the performance was better when including it even thogh I wanted to exclude it since my images after applying only the directional threshold where very noicy. 

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform includes a function called `warp()`, which appears in cell 14 through 18 in the file `main.ipynb`.  The `warp()` function takes as inputs an image (`img` and `perspective_transform_matrix`), to find the perspective transform matrix I make a call to `cv2.getPerspectiveTransform` using source (`src`) and destination (`dst`) points.  I chose the hardcode the source and destination points in the following manner:

```python
bottom, _ = (img.shape[0] - 1, img.shape[1] - 1)
pts = np.array([[210,bottom],[595,450],[690,450], [1110, bottom]], np.int32)
src = pts.astype(np.float32)
dst = np.array([[200, bottom], [200, 0], [1000, 0], [1000, bottom]], np.float32)
```

This resulted in the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 210, 719      | 200, 719      | 
| 595, 450      | 200, 0        |
| 690, 450      | 1000, 0       |
| 1110, 719     | 1000, 719     |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![alt text][image4]

#### 4. Describe how you identified lane-line pixels and fit their positions with a polynomial?

I started with some experimenting in cell 20 and then the final implementation in cell 21 in `main.ipynb`.
To solve this problem i first cut the binary warped image in half in y direction. Then I summed up all pixels colums assuming that the colum having the highest sum would represent the lines. After this I identified the argmax for the left and right half of the image to find both lines. Using the identified peek values as the center values I could check all pixels within a box around that area to identify all white pixels. If I could find more then 50 pixels I would move the bounding box up one level and create the new center as the mean of all mixels found in the last box. When reaching the top of the image I have collected all pixels representing the lines. I can not send all the pixels for the right and left lanes into numpys `polyfit` with a second order argument to get back the coefficients representing the best fix to the pixels for a curved line.
Here is an exampel image where the slinding windows are drawn and all pixels indentified in the left and right lane are highlighted:

![alt text][image5]

For an image where a smoother line is fitted check the next image or the over lay images shown in the video.

#### 5. Describe how you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

I did this in cell 21 in function `_calculate_curvature` my code in `main.ipynb`.
I did this by lookinging for the radius of the smalles circle tangent to the lane line as found in our polynomial fit. I also needed to convert from pixel values to real world measures.
There is a very good [article](https://www.intmath.com/applications-differentiation/8-radius-curvature.php) for more information.
I used to following formula to calculate the radius.
```python
ym_per_pix = 30/720 # meters per pixel in y dimension
xm_per_pix = 3.7/700 # meters per pixel in x dimension
r = (1 + (first_deriv)**2)**1.5 / abs(second_deriv)
```

For calculating how much the car was off from the center I removed the average starting point for the left and right ling, then subtracting the middlepoint and then finding out the pixel to real workind with ratio I could multiply using that,

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in cell 21 in my code in `main.ipynb` in the functio `_draw_lanes()`.  Here is an example of my result on a test image:

![alt text][image6]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./output_video/challange.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

This was by far the hardes task so far. What made it so hard was the fact that the thresholding was not very robust. It worked ok for some frames and then something happened and things stoped working and I had to try different thresholds. So I would say the two most critical point are the thresholding and the perspective transform. If for example the camera move or if the road goes uphill or downhill the perspective transform will break.
