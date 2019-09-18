## Writeup Advanced Lane Finding Project

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

[image1]: ./output_images/test_calibration_example.png "Undistorted"
[image2]: ./output_images/calibration_example.png "Road Transformed"
[image3]: ./output_images/gradient_and_color_threshold_example.png "Gradient and Color Threshold"
[image4]: ./output_images/binary_image_example.png "Binary Example"
[image5]: ./output_images/image_warp_example.png "Warp Example"
[image6]: ./output_images/sliding_window_fit_example.png "Sliding Window Fit"
[image7]: ./output_images/classification_and_fit_example.png "Prior Poly Fit"
[image8]: ./output_images/final_output_example.png "Prior Poly Fit"
[video1]: ./output_images/project_video.mp4 "Video"
[video2]: ./output_images/challenge_video.mp4 "Challenge Video"
[video3]: ./output_images/harder_challenge_video.mp4 "Harder Challenge Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points  

---

### Camera Calibration

The code for this step is contained in the python module "camera_calibration.py" in the function "cal_cam".

I start by preparing "object points", which are the (x, y, z) coordinates of the chessboard corners in the world.
Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each
calibration image and are just copied on a successful detection. `imgpoints` will be appended with the
(x, y) pixel position of each of the corners in the image plane on a successful chessboard detection.  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using
 the `cv2.calibrateCamera()` function.  Below you can see an example output, when applying the `cv.undistort()` function
to a test image with the calculated parameters (in the module "camera_calibration.py" this is encapsulated in the
function `undist_image` for code readability)

![alt text][image1]

### Lane Finding Pipeline

Before the start of this pipeline, the above mentioned camera calibration has been done successfully.

#### 1. Image undistortion

At the beginning of the pipeline the images are undistorted using the `undist_image` function located in
`camera_calibration.py` line #53 to undistort the camera images using the above calculated camera specific values.
Below this step is shown on an example image.
![alt text][image2]

#### 2. Creation of a binary images showing edges

Next, a combination of a color threshold and a gradient detection and threshold is applied to the undistorted image
to filter out any edges in the image. All this is achieved by the function `color_threshold` in the module
`camera_calibration.py`

First, the image is transformed to HLS color space using the function `cv2.cvtColor` of the OpenCV library.
An edge detection using the Sobel operator in x direction on the L color channel is performed. To obtain a binary image,
a low and high threshold of 20 and 100, respectively, is applied to the result of the Sobel-x Operator application on
the image. Another binary image is created by applying a simple highpass filter on the S channel of the HLS image, where
all the values above 120 intensity yield 1. Finally these two binary images are combined with a logical or operation.
Below you can see the application of this function on the undistorted image above.

![alt text][image3]

In this image, the result of the gradient threshold is displayed in green, while the color threshold is shown in blue.
The final binary image is shown here.

![alt text][image4]

#### 3. Perspective Transformation to Birdseye view

Next, a perspective transformation to a birdseye view (== camera above and perpendicular to the ground facing towards
the ground) is performed by the function `warp_to_birdseye()`, which appears in lines 65 through 103 in the
 file `camera_calibration.py`.  The function performs this transformation using hardcoded source `src` and destination
 `dst` points and the OpenCV function `cv2.warpPerspective`. 
The four source points were chosen by optical inspection of an image with straight lines, on the outer borders of the
left and right lane. As these lines should be straight in the birdseye view, a rectangle that scales with the image
parameters was chosen as the four destination points, as shown below in the code snippet:

```python
src = np.float32([[x/2 + 66, y - 260],
                  [x / 2 - 65, y - 260],
                  [x/2 - 410, y - 30],
                  [x/2 + 435, y - 30]])
                  
y, x = img.shape
x_offset, y_offset = 150, 50
dst = np.float32([[x - x_offset, y_offset],
                  [x_offset, y_offset],
                  [x_offset, y - y_offset],
                  [x - x_offset, y - y_offset]])
```

These source and destination points are shown below overlaid over the original and transformed binary image of the
lines:

![alt text][image5]

#### 4. Fit of two Second Degree Polynomials as Left and Right Lane

Finally the warped binary image in birdseye view is used to fit the left and right line on the ego lane using second
degree polynomials. The biggest issue here is to decide which pixels belong to the left or to the right line,
respectively. For this purpose there are two distinct methods implemented in the module `lane_finder.py`. If there is 
nothing known about the lines, a few sliding windows are used to classify and fit the two lines. If there has been a 
previous successful fit, the pixels are classified and fit again based on the previous fit.

The sliding window approach is found in the function `sliding_window_search()` in lines 23 - 119 of the module
`lane_finder.py`. Here the left and right pixels are classified based on a histogram over the y-axis of the bottom half
of the image, choosing the two highest peaks as a starting point.
```python
histogram = np.sum(binary_warped[binary_warped.shape[0] // 2:, :], axis=0)
midpoint = np.int(histogram.shape[0] // 2)
leftx_base = np.argmax(histogram[:midpoint])
rightx_base = np.argmax(histogram[midpoint:]) + midpoint
```
Then 9 subsequent sliding windows are used to fully classify the pixels belonging to the left and right lane, where
each sliding window has a width of 50 pixels and updates the position of the following one by taking the average
x-position of all pixels contained in it, to guarantee that the windows follow the curvature. When all the pixels are
classified in left and right (or unclassified!), two polynomials of second order are fit to these pixels using linear
regression with the method of least squares.

The exemplary output of this algorithm is shown on the warped binary image below:

![alt text][image6]

The other, computationally more simple and less runtime intensive method is using previous fits to classify the
pixels. This method is implemented in `search_around_poly()` in lines 122 - 184 of the module `lane_finder.py`. Here
the two given fits are used to calculate an area around them, with a margin of 100 pixels to each side, and simply
choose the pixels contained in them to classify the left and right line pixels. 

When all the pixels are classified in left and right (or unclassified!), two polynomials of second order are fit to
these pixels using linear regression with the method of least squares.

The result is shown on the same warped binary image below:

![alt text][image7]

#### 5. Calculation of Lane Radius of Curvature and Camera Offset

As a last step, the lane radius of curvature and camera offset are calculated in the method `measure_lane_geometry()`
of the class `LaneFinder()` in the lines 234 - 271 of the module `lane_finder.py` (description of class see chapter
Pipeline).
In this method pixels are converted to meters based on previously measured calibration values.

In this method the radius of the left and right lane is calculated based on the polynomial parameters according to:
```python
left_curverad = (1 + (2 * self.left_fit[0] * y_eval + self.left_fit[1]) ** 2) ** (1.5)
                / (2 * np.abs(self.left_fit[0]))
right_curverad = (1 + (2 * self.right_fit[0] * y_eval + self.right_fit[1]) ** 2) ** (1.5)
                 / (2 * np.abs(self.right_fit[0]))
```
as described in the lessons. As an evaluation point for the radius of curvature, the pixel closest to the car has been
chosen (maximum image y-value). These two values are averaged to yield the lane radius of curvature.

Additionally the camera offset is calculated by simply using the midpoint in x of the left and right polynomials
and comparing it to the image midpoint x-value:
```python
left_fitx = self.left_fit[0] * y_eval ** 2 + self.left_fit[1] * y_eval + self.left_fit[2]
right_fitx = self.right_fit[0] * y_eval ** 2 + self.right_fit[1] * y_eval + self.right_fit[2]
offset_to_lane = (((left_fitx + right_fitx) / 2.) - np.max(img.shape[0])) * xm_per_pix
```
These values are then projected as text on the unwarped image, which is described in the next section.

#### Final Display of the Results

The lane is visualized by a polygon within the method `find_lane()` of the class `LaneFinder()` in the lines 218 - 228
of the module `lane_finder.py` (description of class see chapter Pipeline), right after the call of the appropriate fit
method. Here the x and y pixels of both polynomials are calculated by evaluating the polynomial over the whole image.
A polygon is calculated between these two pixels using the OpenCV function `cv2.fillPoly()` and filled with green pixels.
Afterwards this image is simply unwarped using the inverse transformation matrix from the birdseye transformation, as 
can be seen in `warp(lines, Minv)` line 22 in `lane_detection_pipeline.py`. This result is then overlaid over the input
image to produce a visualization of the lane as can be seen in the image below.

![alt text][image8]

---

### Processing Pipeline

To finally run on a video stream of images, the pipeline used in `lane_find_on_image()` of the module
`lane_detection_pipeline.py` has to be extended with respect the above steps. This is needed to deal with unvalid fits
due to unclear camera images and resulting errors in the pipeline.

Namely we introduce a class `LaneFinder`, found in line 183 of the module `lane_finder.py`, which is instantiated before
executing the lane detection pipeline. This class will perform the following tasks:
- Storing the previous 5 fit results:
    - For this the class has two fit container attributes, `LaneFinder.left_fit_hist` and `LaneFinder.right_fit_hist`,
    which are filled from the front after each frame.
- Based on the history, calling the correct fitting method, search from prior fit or sliding window search:
    - The method `LaneFinder.check_history()` determines which fit method shall be used.
    - If the last three fit attempts have been invalid, we will use the sliding window search, otherwise we search based
    on the last smoothed fit.
- Determining if a fit can be deemed valid, e.g. makes sense under the assumption we are looking at the ego lane:
    - This is performed in the method `LaneFinder.write_history()`, which is called after each fit run.
    - In this method current fit parameters are compared and add to a confidence value if they fit the expectations.
    Namely, we expect the curvature and slope of left and right fit to be within 50% of each other, and we expect the 
    lane width not to be below beneath 500 pixels. Additionally we penalise if the lane width is too low to guarantee a
    reset of the fitting mechanism when the lane width is too low.
    - Finally if the confidence value is above 2 the fits are added to the history, otherwise an invalid fit is entered
    ([None, None, None]) in the history container, if it is not already full of invalid fits.
- Averaging over all valid fits in the history to prevent sudden lane jumps:
    - In the method `LaneFinder.smooth()` all the valid fit parameters are averaged and written to the current
    `LaneFinder.left_fit` and `LaneFinder.right_fit` class attributes.
    
The correct execution of these steps (along with the lane visualization) is done by a call of the function
`LaneFinder.find_lane()`, which is called in the `lane_find_on_image()` at the correct stage.

Using this pipeline, we are able to produce this [video][video1], which demonstrates the capabilities of
the algorithm.

---

### Discussion

Unfortunately, the provided pipeline does not perform well with the two additional challenge videos, which can be found
[here][video2] and [here][video3].

The reason for this is on the one hand the pixel classification and fitting mechanisms. The provided pixel
classification methods are highly vulnerable when there are other edges detected close to the lane lines, as for example
the tar stripes in the [second video][video2]. This could be prevented by a more sophisticated pixel
classification, for example variable window sizes based on pixel history. Obviously we could also work on more carefully
selecting edge pixels, using a more complex filtering than simple high or bandpass logic, or applying more color channels.

Another large issue is, as can be seen in the last [challenge video][video3] for example, when the curvatures become to
narrow, or if we would even have to deal with lane crossings. Here, we would need to think about an adaptive region interest or
similar approaches. At this stage it might also become obvious, that polynomials of second degree are not the ideal
functionals to express real road lanes because they are not complex enough. One can either
increase the degree of the polynomial to a certain extend, use a different representation like Bezier curves or even use
a spline approach, which would largely complicate the fitting mechanism.

Obviously, our tracking and history checks can also be improved. We could provide a more sophisticated calculation of
the sanity checks that for example also relies on what happened in the earlier history. A big benefit could certainly gained
here by the usage of an extended Kalman filter or similar tracking mechanisms, which we will learn more about in a few
lessons!
