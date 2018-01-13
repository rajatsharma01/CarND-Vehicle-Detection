## Vehicle Detection Project

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./examples/car_not_car.png
[image2]: ./examples/HOG_example.jpg
[image3]: ./examples/sliding_windows.jpg
[image4]: ./examples/sliding_window.jpg
[image5]: ./examples/bboxes_and_heat.png
[image6]: ./examples/labels_map.png
[image7]: ./examples/output_bboxes.png
[video1]: ./project_video.mp4
[Vehicle]: ./examples/VehicleExamples.png
[NonVehicle]: ./examples/NonVehicleExamples.png
[FinalOutput]: ./examples/FinalOutput.png
[GammaCorrection]: ./examples/GammaCorrection.png
[HOGFeatures]: ./examples/HOG_feature.png
[HeatmapLabelBox]: ./examples/HeatmapLabelBox.png
[HeatmapLeftWall]: ./examples/HeatmapWithLeftWall.png
[HistEqualGamma]: ./examples/HistogramEqualizationGammaCorrection.png
[HogSubsample]: ./examples/HogSubsamplingSearchOutput.png
[Darkness]: ./examples/MeanPixelValue.png
[Normalization]: ./examples/Normalization.png
[SlidingWindowSearch]: ./examples/SlidingWindowSearchOutput.png
[SlidingWindows]: ./examples/SlidingWindows.png
[ProjectVideoOutput]: ./output_images/project_video.mp4
[TestVideoOutput]: ./output_images/test_video.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Vehicle-Detection/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contained in the 3rd code cell of the IPython notebook. I started by reading in all the `vehicle` and `non-vehicle` images.  Here are some examples of the `vehicle` and `non-vehicle` classes:

![Vehicle]
![NonVehicle]

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an example using the `YCrCb` color space and HOG parameters of `orientations=9`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:

![HOGFeatures]

More examples are present in output of the code cell 3 in notebook.

#### 2. Explain how you settled on your final choice of HOG parameters.

I had experimented with HSV, HLS, YUV, LUV and YCrCb color spaces with different orientation and pixel_per_cell combinations and finally settled on following set of parameters which gave best test accuracy with SVM classifier:

`
# Parameters to extract features
color_space = 'YCrCb'
spatial_size=(16,16)
hist_bins=32
orient = 9
pix_per_cell = 8
cell_per_block = 2
hog_channel = 'ALL'
spatial_feature = True
hist_feature = True
hog_feature = True
`

#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

Code cell 4 has methods to extract color featues (histogram and spatial) and cell 5 has interface to go over set of images and extract combined hog and color features. Code cell 7 extracts features from training dataset [vehicle](https://s3.amazonaws.com/udacity-sdc/Vehicle_Tracking/vehicles.zip) and [non-vehicle](https://s3.amazonaws.com/udacity-sdc/Vehicle_Tracking/non-vehicles.zip) images. Each image is flipped horizontally as well to double the number of examples. Features are normalized using `sklearn.preprocessing.StandardScaler`. This is how features vector looks like before and after normalization:

![Normalization]

Features are split in training and test data sets using `sklearn.model_selection.train_test_split` and fed into LinearSVM classifier in cell 11. Classifier achieves test accuracy of *98.7%*.

### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

I have used 3 regions for sliding windows in y ranges of (390, 630), (400, 570) and (400, 520) with sliding window size of 96, 80, 64 respectively with an window overlap of 85%. *Sliding Window Search* section in project notebook (cells 17 and 18) implement this.  

![SlidingWindows]

I have used HOG subsampling approach to optimize calculating HOG features only once for the whole search region. I have used 3 scales corresponding to above window sizes i.e. 1.5, 1.25 and 1.0. To increase overlap between windows, I have set `cells_per_step` to 1, which gives an overlap of 87.5%. *HOG Sub-sampling Window Search* (cell 19) implements this appoach.

#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

I have used some preprocessing of the images to enhance performance of the classification. Some of the frames in project video are darker due to tree shadows on the road. To Quantify darkness, I have calculated mean pixel value of grayscale image for each frame (code cell 14). Following example shows mean pixel values for test image:

![Darkness]

Next, I have used [Gamma Correction](https://www.pyimagesearch.com/2015/10/05/opencv-gamma-correction/) to fix shadows in these frames. Correction is adaptive to darkness (mean pixel value) of the image, darker image is corrected with gamma factor of 4 vs. less darker image is corrected with gamma of 1.5, lighter images are left untouched. Code cell 15 implements it. Here are examples of Gamma correction:

![GammaCorrection]

Next I have used [Contrast Limited Adaptive Histogram Equalization](https://docs.opencv.org/3.1.0/d5/daf/tutorial_py_histogram_equalization.html) to improve contrast of the images. Below example shows histogram equalization for one of the test image. There are two choices to apply Gamma before or after equilization. It turns out to be better to use equilization before and Gamma correction after for classifier to work better (Image also looks better that way).

![HistEqualGamma]

Ultimately I searched on three scales (1.5, 1.25, 1.0) using YCrCb 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, here are some example output images with detection windows:

![HogSubsample]

There are many overlapping windows over detected cars and false positives. I have used heatmaps and `scipy.ndimage.measurements.label()` to identify bounding boxes around detections. By thresholding heatmap with lower values, I have reduced false positives. This is an example of using heatmaps to find bounding boxes:

![HeatmapLabelBox]

The entire project video runs in leftmost lane with a barrier on its left side. In many frames, my classifier can catch cars from reverse direction, we probably are not interested in locating those. Since detecting a barrier itself might be a project of its own, I relied on lane dection module to give me left line polynomial with a margin (50 pixels) in unwarped space of original image. `LaneDetector.get_unwarped_line_fit` of `AdvanceLaneFinding.py` implements this. With this polynomial, I can now create a virtual boundary wall and filter out any windows crossing this wall boundary. This is implemented by `add_heat_with_left_wall` in code cell 21. This is how detections look like with left wall:

![HeatmapLeftWall]

---
### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a link to output of my pipeline on [![project_video](http://img.youtube.com/vi/z_bSm2S1b2E/0.jpg)](http://www.youtube.com/watch?v=z_bSm2S1b2E)

Note that the detection for white car starts of little late util camera starts seeing tail portion of the car, this could be attributed to lesser number of examples for similar white cars. I tried to augment dataset with Autti data, but my Ipython notebook sesson kept running out of memory after loading data for about 30 minutes. Also LinearSVM does not work with generators, it requires all the data points in memory.

Additionally, if I lower heatmap threshold values, it detects white car better at lower thresholds, but that also opens up many falls positive detections. Also it takes more number of frames for white car heamap to heat up to cross this threshold. Suggestions to improve it further are highly appreciated.

#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.

The final pipeline is implemented in code cell 22. I have used a `deque` of heatmaps from last 10 frames to identify car positions. I use average value of these 10 frames heatmaps to find out positions of cars. This helps with identifying cars with higher confidence and reduce false positives which only appear in 1 or 2 frames. The final project video also displays these average heatmap values after applying thresholds on the top right corner (along with perspective view of lane sliding window search). Its easy to colerate changes to heatmap in video as car moves across these frames.

I have also used lane detection module from Project 4 to identify lane mask and curvature metrices, which are displayed over final output image along with detected cars bounding boxes. Here is how output looks like for test images:

![FinalOutput]

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

In this project, I have used HOG features + Sliding windows search with SVM classifier to identify car images as described in lessons for this project. I did find myself struggling to fine tune the parameters and make them work for the project videos. Still I feel my implementaion is not yet robust to generalize well. Here are the challenges I felt:

1. Limited memory to load larger dataset as SVM doesn't work with generators.
2. Many parameters to fine tune.
3. It takes about more than an hour for my pipeline to process whole project video on my laptop with no GPU.

I like some other implementations which still rely on HOG + SVM combination, yet achieve marvelous results e.g. project by [John Chen](https://www.youtube.com/watch?v=lryYC6wgVpA) uses voxelization techniques to find bounding 3D boxes around the car from 2D images alone is a masterpiece!

There is another dimension of using deep learning approach to figure out bounding boxes by itself which tend to result in better performance, e.g. YOLO, Mask RCNN based implementations as some other students have done.

What if we can combine these two together to auto detect Voxel bounding boxes around 2D image objects using deep learning models? [VoxelNet](https://arxiv.org/pdf/1711.06396.pdf) exactly does that and I am curious to try this out in future.

