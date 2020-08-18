## Writeup Template

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

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

[image1]: ./examples/undistort_output.png "Undistorted"
[my_image_1]: ./output_images/corner_found_0.jpg "Undistorted1"

[my_image_1b]: ./output_images/calibration_output_6.jpg "Undistorted and Warped Image"
[my_threshold_result]: ./output_images/pipeline_result.jpg "threshold result"
[my_warped_result]: ./output_images/warped_result.jpg "Warped result"

[test_image_0]: ./output_images/lane_detected_examples_output_0.jpg "Detected_lanes_0"
[test_image_1]: ./output_images/lane_detected_examples_output_1.jpg "Detected_lanes_1"
[test_image_2]: ./output_images/lane_detected_examples_output_2.jpg "Detected_lanes_2"
[test_image_3]: ./output_images/lane_detected_examples_output_3.jpg "Detected_lanes_3"
[test_image_4]: ./output_images/lane_detected_examples_output_4.jpg "Detected_lanes_4"
[test_image_5]: ./output_images/lane_detected_examples_output_5.jpg "Detected_lanes_5"
[test_image_6]: ./output_images/lane_detected_examples_output_6.jpg "Detected_lanes_6"
[test_image_7]: ./output_images/lane_detected_examples_output_7.jpg "Detected_lanes_7"

[image2]: ./test_images/test1.jpg "Road Transformed"
[image3]: ./examples/binary_combo_example.jpg "Binary Example"
[image4]: ./examples/warped_straight_lines.jpg "Warp Example"
[image5]: ./examples/color_fit_lines.jpg "Fit Visual"
[image6]: ./examples/example_output.jpg "Output"
[video1]: ./project_video.mp4 "Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.

---

### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Advanced-Lane-Lines/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.

You're reading it!

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained in the few code cell of the IPython notebook located in "P2.ipynb".

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result:

corners were found in
![alt text][my_image_1]

![alt text][my_image_1b]

![alt text][image1]

In my project I also used two functions to re-use them in video processing i.e  warper() and cal_undistort()

All the test images can be seen in folder /output_images/


### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

To demonstrate this step, I will describe how I apply the distortion correction to one of the test images like this one:
![alt text][image2]

As we can see in the IPython notebook that we used a raw image to apply different gradients in the image and after that we produced the output of
image as Threshold Grad Dir image as we can see in "# Test threshold filters on image" code

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

I used a warped image to apply threshold filters
I used a combination of color and gradient thresholds to generate a binary image under the heading "use pipeline on wraped to app".
Here's an example of my output for this step.  (note: this is not actually from one of the test images)

![alt text][image3]
![alt text][my_threshold_result]

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

Before using the wraper function on the images i used a function called region_of_interest(img, vertices) which was provided in project-1
This function created gave us the region we want to concentrate on i.e lane lines

vertices for region of interest
```python
	vertices = np.array([[(190,720),(610, 410),(670, 410), (1100,720)]],dtype=np.int32)
```
The code for my perspective transform includes a function called `warper()`, which appears in the 7th code cell of the IPython notebook.  The `warper()` function takes as inputs an image (`img`), as well as source (`src`) and destination (`dst`) points.  I chose the hardcode the source and destination points in the following manner:

```python
src_for_warp = np.array([[(196,720),(600, 450),(680, 450), (1135,720)]],dtype=np.float32)
dst_for_warp = np.array([[(320,720),(320, 0),(960, 0),(960,720)]], dtype=np.float32)
```

This resulted in the following source and destination points:

| Source        | Destination   |
|:-------------:|:-------------:|
| 196,720      	| 320,720       |
| 600, 450      | 320, 0     	|
| 680, 450      | 960, 0      	|
| 1135,720      | 960, 720      |

I verified that my perspective transform was working as expected by drawing the `src_for_warp` and `dst_for_warp` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![alt text][my_warped_result]
![alt text][image4]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

Then I used a histogram to find the left lane pixels and right lane pixels by using the method of sliding window

Prams used in my code
Choose the number of sliding windows
nwindows = 9
Set the width of the windows +/- margin
margin = 100
Set minimum number of pixels found to recenter window
minpix = 50

Then I used the polly fit function on the pixels detected above with a 2nd order polynomial

![alt text][image5]

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

I did this in function measure_curvature_real_center() in the IPython Notebook
by using the formulas:

```Python
ym_per_pix = 30/720
xm_per_pix = 3.7/lane_values
left_curverad = ((1+(2*left_fit[0]*y_eval + left_fit[1])**2)**1.5)/np.absolute(2*left_fit[0])
right_curverad = ((1+(2*right_fit[0]*y_eval + right_fit[1])**2)**1.5)/np.absolute(2*right_fit[0])

 ```
where, lane_values = int(right_fitx[719]-left_fitx[719]) # bottom of the image

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step by using the weighted image function given in the project 1
Here is an example of my result on a test image:

![alt text][image6]
![alt text][test_image_0]
![alt text][test_image_1]
![alt text][test_image_2]
![alt text][test_image_3]
![alt text][test_image_4]
![alt text][test_image_5]
![alt text][test_image_6]
![alt text][test_image_7]


---

### Pipeline (video)

For the complete pipeline, I used the function one_image_processing() and the Line class provided in the example code

complete flow of any image goes like this:

1) First, we undistort the image
2) Then we will apply the threshold and color gradient filters
3) on the gradient image we will apply region of interest filter and the warp the image to bird’s eye view
4) on the warped image we will try to find the right and left lanes pixels
5) for each side of pixels we will fit a 2nd order polynomial if we already have a previously fit coefficient in the Line class object. we will only search the area near the line within the margin range. Also after every 20 frames, we use the histogram to calculate the lines again to make sure that we are still calculating the correct lines and then repeat the process again
6) After calculating all the variables we will add the data in Line class object
7) we Also calculate the average of the lines coefficient that we detected in the last few steps
8) We also calculate the distance from the center, left curvature, right curvature for the detected line
9) Finally, we will draw all the pixels in a new image with the same shape and then use weighted image function to create the final image


#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./test_video_output/clip.mp4)

Here's a [link to my video result for challenge video](./test_video_output/challenge_project_video.mp4)
---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.

The first problem is the sudden change in the path ahead as we can see in the project video at around 22 second's our road suddenly changed but the current video lines are averaged our line didn't change as frequently as we wanted to, hence these are some points where our lines are shaky
Prospective solution:  we need to make sure that every time we calculate the lines ahead we are checking the current output is correct so we can calculate it through 2 methods or a better method from the current one

Second, As I ran my project code on the challenge video I found a flaw in our code
When we have a lane with some constructions and change in color the detected lines are not correct as the construction in the middle of the lane causes our algorithm to detect the wrong path
Prospective Solution:  We need a better approach to remove the unwanted part I the lane detection line not considering the pixels at the center or change in color
