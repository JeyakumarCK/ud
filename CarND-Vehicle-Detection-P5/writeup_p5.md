##**Vehicle Detection Project**
####by Jeyakumar C.K.
---
The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector.
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
###Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Vehicle-Detection/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

P5-Vehicle Detection project writeup is given in the file [writeup_p5.md](writeup_p5.md) file. The code for this project is in [p51.py](p51.py) file.

###Histogram of Oriented Gradients (HOG)

####1. Explain how (and identify where in your code) you extracted HOG features from the training images.

In my program file p51.py, there is a function by name `get_hog_features` which takes an image and other parameters as input and gives out HOG features as an output for the given image. The key input parameters for this method are:
- image: a single image containing 3 channels
- orient: number of bins to split the orientations
- pixels_per_cell: number of pixels to be used in each side of a 2D cell
- cells_per_block: number of cells to be used in each side of of a 2D block
- visualise: a flag to get an HOG Image as an additional output
- feature_vector: a flag to convert the HOG features in to 1D list

In order to train a classifier using training data, first I had to read all the test data.  The function `loadTrainingImages` loads all names of the training image file names in a list and send it to the function `extract_features`. This function reads each file name and loads the image data using `mpimg.imread`.  This image is passed to the `get_hog_features` function that described above.  And the output hog features are collected back in to another array, and used to train the `LinearSVC` classifier instance.  

In order to effectively extract the HOG features from a given image, it is important to choose the right parameters for orientations bin count, pixels per cell, cells per block.  And also we need to choose the right color space and image channels of input image to use for HOG feature generation.  In the below sample images of HOG features (which are randomly chosen from the training image set), I have shown HOG features using different color space and channels.

HOG Feature of Car samples:
![Car: HOG Features of S-Channel of HLS](output_images/hog_car_schannel.png)
![Car](output_images/hog_car_lchannel_1.png)
![Car](output_images/hog_car_crchannel_1.png)

HOG Feature of Not Car samples:
![NotCar](output_images/hog_notcar_lchannel_1.png)
![NotCar](output_images/hog_notcar_vchannel_1.png)
![NotCar](output_images/hog_notcar_vchannel_2.png)

####2. Explain how you settled on your final choice of HOG parameters.

I tried various combinations of parameters and...

####3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I trained a linear SVM using...

###Sliding Window Search

####1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

I decided to search random window positions at random scales all over the image and came up with this (ok just kidding I didn't actually ;):

![alt text][image3]

####2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on two scales using YCrCb 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result.  Here are some example images:

![alt text][image4]
---

### Video Implementation

####1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./project_video.mp4)


####2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

### Here are six frames and their corresponding heatmaps:

![alt text][image5]

### Here is the output of `scipy.ndimage.measurements.label()` on the integrated heatmap from all six frames:
![alt text][image6]

### Here the resulting bounding boxes are drawn onto the last frame in the series:
![alt text][image7]



---

###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  
