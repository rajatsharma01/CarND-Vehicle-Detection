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

Code cell 4 has methods to extract color featues (histogram and spatial) and cell 5 has interface to go over set of images and extract combined hog and color features. Code cell 7 extracts features from training dataset [vehicle](https://s3.amazonaws.com/udacity-sdc/Vehicle_Tracking/vehicles.zip) and [non-vehicle](https://s3.amazonaws.com/udacity-sdc/Vehicle_Tracking/non-vehicles.zip) images. Each image is flipped horizontally as well to double the number of examples. Features are normalized using `sklearn.preprocessing.StandardScaler`. This is how features look like before and after normalization:

![Normalization]

Features are split in training and test data sets using `sklearn.model_selection.train_test_split` and fed into LinearSVM classifier in cell 11. Classifier achieves test accuracy of *98.7%*. I trained a linear SVM using

### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

I have used 3 regions for sliding windows in y ranges of (390, 630), (400, 570) and (400, 520) with sliding window size of 96, 80, 64 respectively with an window overlap of 85%. *Sliding Window Search* section in project notebook (cells 17 and 18) implement this.  

![SlidingWindows]

I have used HOG subsampling approach to optimize calculating HOG features only once for the whole search region. I have used 3 scales corresponding to above window sizes i.e. 1.5, 1.25 and 1.0. To increase overlap between windows, I have set `cells_per_step` to 1, which gives an overlap of 87.5%. *HOG Sub-sampling Window Search* (cell 19) implements this appoach.

#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on two scales using YCrCb 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result.  Here are some example images:

![alt text][image4]
---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](https://youtu.be/z_bSm2S1b2E)


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

### Here are six frames and their corresponding heatmaps:

![alt text][image5]

### Here is the output of `scipy.ndimage.measurements.label()` on the integrated heatmap from all six frames:
![alt text][image6]

### Here the resulting bounding boxes are drawn onto the last frame in the series:
![alt text][image7]



---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  

