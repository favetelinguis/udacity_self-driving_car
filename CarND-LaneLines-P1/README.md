# **Finding Lane Lines on the Road** 

---

**Finding Lane Lines on the Road**

The goals / steps of this project are the following:
* Make a pipeline that finds lane lines on the road
* Reflect on your work in a written report

---

### Reflection

### 1. Describe your pipeline. As part of the description, explain how you modified the draw_lines() function.

My pipeline consisted of 5 steps. First, I converted the images to grayscale, then I 
apply a Gaussian blur to make the less sharp features in the image less eye catching.
Then I apply canny edge detection to find where the gradients (color shifts) are the greatest. A region of interest is created in front of the car to make it easier to find what edges are relevant for the task. Finally a Hough lines transform is performed to
filter our the segments of gradient shifts found in the canny edge detection that does not belong to a whole line. The last step in the pipeline is to draw the lines found in the region of interest over the original image.

In order to draw a single line on the left and right lanes, I modified the draw_lines() function by first separating the negative and the positive slopes of the line segments with the hopes that negative slopes represent the left side and positive slopes represents the right line. 
During the calculation of slopes 3 filters are applied to remove segments that does not belong to the lines of interest. First is the case when dx or dy equals zero since at this point the gradient is undefined. The third filter is for the case when the slope of the line has an slope that is adjacent to what can be expected as a
straight line in front of the car.
After filtering out only the segments that I believe belong to the lane lines I run a
linear regression with first order features to find the slope and intersect of the lines that best approximates the line segments of the left and right side. When I have the slope and intersect I use the straight line function to calculate x given y at the bottom of the
image and at some arbitrary point in front of the car.

### 2. Identify potential shortcomings with your current pipeline

Curved lines, hilly road, arbitrary lines that comes into the region of interest
but that does not belong to the lane lines for example striped colored cars.

### 3. Suggest possible improvements to your pipeline

In order to make a more robust pipeline I think deep learning could help me. If I
could get lots of annotated images showing where the lines are I could train a model to identify the lines. I have also found information online about another general
Hough transform, given that this will give me points belonging to some curved shape I
can then use linear regression with second order features to fit a curved line.