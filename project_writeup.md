##Project 5: Vehicle Detection
###This is a summary of the vehicle detection project work - the approaches taken and the project results.

---

**Vehicle Detection Project**

The goals / steps of this project are the following:

* The **goal** of the project is to detect and track vehicles that are traveling on the road.
* The primary **approach** is to extract features in each video image frame and use them for a Linear SVM classifier to classify if an image window contains a car or not. This SVM classfier is used in the context of a multi-scale sliding window search mechanism to identify the locations and sizes of vehichle within the image. The identified vehicles are then tracked in the video frame sequence.
* The features extracted for the Linear SVM classifier including:
    * Histogram of Oriented Gradients (HOG) features (8x8 pixels per cell, and 2x2 cells per block)
    * Color features (a concatenation of 3 color channels)
    * Color histogram features
* A multi-scale sliding window search for cars is performed on the bottom half of the image to detect vehicles of different scales.
* The detection is run on each video frame, and certain tracking contraints are imposed to filter out possible false detections. The requirement in the tracker is typically the same detection needs to appear in adjacent locations across multiple video frames.
* The qualified detection windows are then overlayed on the output video frames to generate the output video.

[//]: # (Image References)
[image1]: ./examples/bboxes_and_heat.png
[image2]: ./examples/car_not_car.png
[image3]: ./examples/HOG_example.jpg
[image4]: ./examples/labels_map.png
[image5]: ./examples/output_bboxes.png
[image6]: ./examples/sliding_window.jpg
[image7]: ./examples/sliding_windows.jpg
[image8]: ./examples/test_car_img.png
[image9]: ./examples/test_car_hog1.png
[image10]: ./examples/test_car_hog2.png
[image11]: ./examples/test_car_hog3.png
[image12]: ./examples/test_notcar_img.png
[image13]: ./examples/test_notcar_hog1.png
[image14]: ./examples/test_notcar_hog2.png
[image15]: ./examples/test_notcar_hog3.png
[image16]: ./examples/test_heatmap_b1.jpg
[image17]: ./examples/test_heatmap_h1.jpg
[image18]: ./examples/test_heatmap_b2.jpg
[image19]: ./examples/test_heatmap_h2.jpg
[image20]: ./examples/test_heatmap_b3.jpg
[image21]: ./examples/test_heatmap_h3.jpg
[image22]: ./examples/test_heatmap_b4.jpg
[image23]: ./examples/test_heatmap_h4.jpg
[image24]: ./examples/test_heatmap_b5.jpg
[image25]: ./examples/test_heatmap_h5.jpg
[image26]: ./examples/test_heatmap_b6.jpg
[image27]: ./examples/test_heatmap_h6.jpg
[image28]: ./examples/test_multiscale1.jpg
[image29]: ./examples/test_multiscale2.jpg
[image30]: ./examples/test_multiscale3.jpg
[image31]: ./examples/test_multiscale4.jpg
[image32]: ./examples/test_multiscale5.jpg
[image33]: ./examples/test_multiscale6.jpg
[video34]: ./project_video.mp4
[code]: ./P5-Vehicle-Detection.ipynb

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
###Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Vehicle-Detection/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

**All code is contained in the jupyter notebook file [P5-Vehicle-Detection.ipynb](https://github.com/xingjin2017/CarND-Vehicle-Detection/blob/master/P5-Vehicle-Detection.ipynb)**

Here's a [link to my video result with a cross-frame tracker](https://youtu.be/9ZCsaPU8P_g)

Here's a [link to my video result without a cross-frame tracker](https://youtu.be/jeXI5l_FT_4)

###Histogram of Oriented Gradients (HOG)

####1. Explain how (and identify where in your code) you extracted HOG features from the training images.

There is a function get_hog_features defined in P5-Vehicle-Detection.ipynb that returns the hog features and optionally the hog feature image for a particular color channel of an image.

I started by reading in all the `cars` and `notcars` images, 8000+ for each category. I explored the color spaces like RGB, HSV, LUV, and YCrCb. Both HSV and YCrCb work well. After comparison, to use all three channels give better results, so all HOG features on the three color channels are used and concatenated together. In the final run, the color space YCrCb is used.

Here is one car image and the hog feature images on the three color channels:

![alt text][image8]![alt text][image9]![alt text][image10]![alt text][image11]

Here is one non-car image and the hog feature images on the three color channels:

![alt text][image12]![alt text][image13]![alt text][image14]![alt text][image15]

####2. Explain how you settled on your final choice of HOG parameters.

I settled with the color space of YCrCb with all three color channels as explained above. pix_per_cell is 8 and cell_per_block of 2 as they perform well enough. orient used 18 instead of 9, as this seems to give higher classification accuracy.

####3. Additional features extracted.

Also extracted the spatial features, by reducing the image to 32x32 and concatenate the values of the three color channels.

On top of this, added color channel histograms for all three color channels, with number of bins as 64. The three channel features are concatenated together.

Before feeding into the classifier, all three types of featues (HOG, spatial, and color histogram) are concatenated together to form one feature vector.

####4. Describe how (and identify where in your code) you trained a classifier using your selected features.

I trained a linear SVM in the train_svc function. The feature vector is normalized using the StandardScaler. All data are randomized and used train_test_split to do a split, 80% for training and 20% for testing. There are about 16000 or so images together, with about 8000+ for cars and 8000+ for non-cars. It takes only a few seconds to train the SVM classifier and the test accuracy is above 99%.

Feature vector length for one image: 13848
7.53 Seconds to train SVC...
Test Accuracy of SVC =  0.9913

One thing noteworthy is the extract_training_image_features function that extract features for all 16000+ images and the extracted are written to local files, so these files are used as caches to avoid recalculating features unnecessarily.

###Sliding Window Search

####1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

multiscale_find_cars is the entrance function that handles the sliding window search. It search for three scales: 1.2, 1.8, 2.7. These translate to 76x76, 115x115 and 172x172 windows. The ranges of searches are different for each scale, 400 to 500 for 76x76, 400 to 580 for 115x115, and 400 to 660 for 172x172. Mostly smaller cars would appear near the horizon and far away, while larger cars can appear anywhere depending on the position. The range limitations are to avoid the unnecessary computations and also to avoid the false positives from them.

The one scale function is find_cars_with_scale, which is called by multiscale_find_cars multiple times.

Here are some examples of the multiple scale search, including with cars and without cars:

![alt text][image28]
![alt text][image29]
![alt text][image30]
![alt text][image31]
![alt text][image32]
![alt text][image33]

####2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched for three scales for better accuracy and used HOG sub-sampling in the lecture to speed up the search to a degree. Also tried four scales search, but in general, the smaller the window size, the more expensive it is to search for it on the image. Three scales would just use half of the time, compared to four scales search. The three scales search and the sizes of the search windows were chosen mostly for accuracy and reducing false positive detection.

One important thing to mention is for the HOG sub-sampling based search, I used cells_per_step of 1, which means 7/8 (**87.5% overlap rate**). For a valid car, this would give a lot of overlapping detections and give high value on the heatmap. I used a **heatmap threshold of 2**. This **combination** leads to accurate detection of cars on the image, with **almost no false positives**.

---

### Video Implementation

####1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)

Here's a [link to my video result with a cross-frame tracker](https://youtu.be/9ZCsaPU8P_g)

Here's a [link to my video result without a cross-frame tracker](https://youtu.be/jeXI5l_FT_4)


####2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

Please refer to the BBoxTracker class in the [jupyter notebook code](https://github.com/xingjin2017/CarND-Vehicle-Detection/blob/master/P5-Vehicle-Detection.ipynb).

In BBoxTracker, it maintains a list of detected car bounding boxes and the latest frame ids that they appeared in. A bounding box is considered to reappear in a new frame, if there is enough overlap (80%) between two bounding boxes. If a bounding box doesn't reappear after box_drop_after_frames (3) frames, it is dropped. If the bounding box has appeared for appear_draw_after_frames (5) number of times, it will be actually drawn:

bboxTracker.reset(overlap_threshold=0.8, box_drop_after_frames=3, appear_draw_after_frames=5)

These thresholds are mostly to prevent false positives. However, I don't find these to be quite useful. When the detection is not very accurate, the same false detection would appear no matter how the threshold is set (unless willing to use a very high threshold of above 30 frames, say, one second of video, which would lead to significant delay in detection).

What worked in the end is to use a high overlap sliding window search (87.5%) and a heatmap threshold of 2. This combination as mentioned has almost no false positive detection in the project vidceo. Given the accuracy, without using a tracker is actually better, because it picks up the fast passing cars from the opposite lanes quickly as well.

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

### Here are six frames and their corresponding heatmaps:

![alt text][image16]
![alt text][image17]
![alt text][image18]
![alt text][image19]
![alt text][image20]
![alt text][image21]
![alt text][image22]
![alt text][image23]
![alt text][image24]
![alt text][image25]
![alt text][image26]
![alt text][image27]

---

###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

The main issues I had or have are the following during the project:

1. False positive handling - if the detection is not accurate, the false positive would appear in the video wihtout a very high detection threshold. It is a bit surprising, given the SVM has 99% accuracy and it would give these false detections. The consistency of the false detection maybe explained by the consistency of the guardrail, for example. In the end, I used high overlap sliding window and a heatmap threshold of 2 to get rid of the false positives.

2. The pipeline speed - it takes about 15 minutes to run the pipeline for the project video, which is not real time at all. This lays bare the inefficiency of sliding window search, probably, although it can be reduced somewhat by combining lane detection and narrowing dow the search ranges.

