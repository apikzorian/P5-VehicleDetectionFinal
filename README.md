# P5 - Vehicle Detection


## Synopsis
In `Project #5 - Vehicle Detection and Tracking`, we used computer vision and classifiers to develop a model that is able to detect cars in an image and track the car through a series of images. Using a linear SVC classifier trained on thousands of images of cars and non-cars, our model was able to detect vehicles on the road and draw a box around its recognized candidate. Once we tried this algorithm on a video, we were able to enhance our model to detect and track a car through its life-span in the video, while rejecting "false positives" in the video as well. 



## Feature Extraction
(Code for this section can be found in `extract_features()` in `lesson_functions.py` on line 51)
We began by applying feature extraction techniques to our data. Our data was provided by Udacity, and featured about 8,000 images, split up into vehicles and non-vehicles. Vehicles images can be found [here](https://s3.amazonaws.com/udacity-sdc/Vehicle_Tracking/vehicles.zip), while non-vehicle images can be found [here](https://s3.amazonaws.com/udacity-sdc/Vehicle_Tracking/non-vehicles.zip) . Below are two examples of vehicle and non-vehicle training images, respectively:


![alt tag](https://s24.postimg.org/6di7d6wfp/image0944.png)

![alt tag](https://s27.postimg.org/noj7ulb9v/image3828.png)


We used `mpimg.imread(file)` to read in our training images. These images were .png files, and since the images we would be testing on are .jpg files, we needed to adjust convert our data to 0-255 space, to be compatible with .jpg test images
                                                `image = (image * 255).astype(np.uint8)`

        
We used three different methods for feature extraction: Color Transform,  Binned Color, and Histogram of Oriented Gradients (HOG). Here are the parameters we set for the feature extraction:

                              ```
                              color_space = 'YUV'  # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
                              orient = 9  # HOG orientations
                              pix_per_cell = 8  # HOG pixels per cell
                              cell_per_block = 2  # HOG cells per block
                              hog_channel = "ALL"  # Can be 0, 1, 2, or "ALL"
                              spatial_size = (16, 16)  # Spatial binning dimensions
                              hist_bins = 16  # Number of histogram bins
                              spatial_feat = True  # Spatial features on or off
                              hist_feat = True  # Histogram features on or off
                              hog_feat = True  # HOG features on or off
                              ```

### Color Histogram and Spatial Binning
(code for this section can be found in `bin_spatial` on line 29 of `lesson_functions.py`)
We played around with a few different color channels, but in the end, saw our best results when we ran chose "YUV" color channel. `feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)`
 
Using `cv2.resize()`, we were able to use spatial binning to scale down the resolution of our image. Using the function `bin_spatial(img, size=(32, 32))`, we were able to resize our training images to 32 x 32 sized images and ravel them to create a feature vetor. By turning our image into a vector, it would then make it easier to feed it into our classifier. `spatial_size = (16, 16)`

We also used histograms of pixel intensity to extract features of our training data. Numpy's `np.histogram()` proved to be very useful in our previous project, when we were trying to detect pixel intensity to detect two lanes on a road, and it was again used here to detect color histogram for vehicles. In `color_hist()`, we computed the color histograms of Red, Blue, and Green, and then concatenated them into a single vector. To keep an orderly bin size, we originally chose to set `hist_bins = 32`. During testing of our pipeline, we realized that reducing the # of bins to 16 actually helped reduce the number of false positives we detected (more on that later!). 

### Histogram of Gradients (HOG)
(code for this section can be found in `get_hog_features` on line 8 of `lesson_functions()`)

Rather than taking the gradient magnitude and direction of each pixel of an image, HOG groups the values into small cells and within these cells, computes the histogram of gradient orientations from each pixel. The gradient samples are distributed into a set of orientation bins and summed up. When observing this histogram, it is important to remember this is not a sum of the number of samples in each direction, but rather a sum of the gradient magnitude of each sample. This reduces the influence of small, noisey gradients, while increasing the weight and influence of stronger gradients. With this applied for all the cells, an image acquires a "signature" of its shape, which is the Histogram of Gradients. 

The biggest advantage of using HOG is that we are able to keep this "signature" distinct enough, while accepting minimal variations in the shape itself. The parameters available to us help change how sensitive these features are. Using the scikit-image package built-in function to extract HOG features (documentation can be found [here] (http://scikit-image.org/docs/dev/api/skimage.feature.html?highlight=feature%20hog#skimage.feature.hog) ), here are the parameters we used for HOG:

                              ```
                              orient = 9  # HOG orientations
                              pix_per_cell = 8  # HOG pixels per cell
                              cell_per_block = 2  # HOG cells per block
                              hog_channel = "ALL"  # Can be 0, 1, 2, or "ALL"
                              ```

`orient` represents the number of bins we would use in the histogram of gradients. 9 was the number we had used in our lessons, and since our images were not very large, it seemed that anything higher would probably be excessive. 

`pix_per_cell` reperesents the number of pixels for each cell of the image we would be running our histogram over. Typically, these cells are chose to be squares, and this value would be (8,8), since pix_per_cell is passed as a tuple.

`cell_per_block` specifies the local area that will be normalized over for a cell's histogram count. This value is also passed as a tuple. Since our images are 64x64 and we are taking 8 pixels per cell, we decided to stick with 2 cells/block (meaning a square 2x2 cell) to keep the measurment balanced.

`hog_channel` specifies which channel we want to extract colors from. This can be either channel 0, 1, 2, 3, or ALL. We experimented with different color channels and in the end, decided on using ALL of them, as this gave us the most vehicles detected on our images.


In `extract_features`, we combine all of these feature extraction methods and apply them to each image. Once all have been applied on an image, these features are concatenated and added to a `features` list, which includes featurs for each image. We do this for both the `cars` and `notcars` lists, which contain training images of vehicles and non-vehicles, respectively. Once we have extracted these features, we combine them in a variable `X` and then create our array of labels, `y`. This array will be filled with 1s for every vehicle feature, and 0s for every non-vehicle feature. 

## Training the Classifier
(code for this section can be found in `P5_video.py`, in the `get_classifier()` function, starting at line 198)
In the function `classify`, we take the features we have extracted from our training data, along with our labels, and train our classifier. We used a type of Support Vector Machine(SVM) classifier. SVMs are supervised learning models that can be used to analyze and classify data. The advantage of SVMs, other than their effectiveness in high dimensional spaces, is that we can specify different kernel functions tailored to our decision making process. For our model, we chose a `sklearn.svm.LinearSVC`. 

We started by fitting a `StandardScaler` to our features to get a per-column fit and shuffling our features and labels to help randomize our data set. We used the `train_test_split` function we have been using in previous projects, to segment some of our training features and labels into testing features and labels. With our testing data in hand, we were ready to test our our LinearSVC. We used the `LinearSVC.fit()` function to fit our model to our training features and labels and then tested the accuracy on our testing features and labels. After playing with some of the features, we were able to consistently get ~98% accuracy on our SVC, with which we were comfortable.

## Searching Windows

### Sliding Windows
(code for this section can be found in `P5_video.py`, in the `get_windows()` function, starting at line 137 and in the `slide_window` function in `lesson_functions.py`)
In our previous project, we had utilized the sliding windows method to help identify lane lines (concept explained here). Our task was a bit more complex now, because rather than identifying peaks in histograms of pixels, we would need to identify actual images of cars and decipher them from non-cars. Luckily, we are equipped with a large supply of car and non-care features, and a classifier that has been trained on them!

In `slide_windows`, we give the image and the parameters of the windows we want to use to search the image for detections. These include where we will start and stop searching in the x direction (`x_start_stop`), the y direction (`y_start_stop`), the size of our window (`xy_window`), and the overlap fraction we want in the windows we are going to draw (`xy_overlap`). With these parameters, we compute the number of windows we want to lay out onto this image and after computing the x,y positions of each window, we return a list that contains the windows. In `get_windows`, we call this function for 4 sizes of windows: tiny, small, medium, and large, with each one varying in window size, as well as start and stop positions in x and y. The idea here was that since objects that are further away seem smaller, we decided to keep the smallest windows closeset to the horizon, while the larger windows closer to the bottom of the image. We also played around with the positions of the windows' to maximize overlap, so that we would be able to have more detections from a varying cascade of windows. Finally, all of are windows covered only the bottom half of the image, since all we care about is the road and anything else would cause false detections. Below is an image with the 4 sliding windows drawn on it:

![alt tag](https://s30.postimg.org/3pfk9uvg1/sliding_windows.png)


Large windows are in black, medium are in red, small are in green, and tiny are in blue


### Searching for Detections
(code for this section can be found in `search_windows` and `single_img_features` on from line 41-137 in `P5_video.py`)

Now that we have our windows, we call the `search_windows` function to extract features from our image using these windows. For each window, we extract its window size and placement from the original image, resize it to 64x64, and then extract the features from that select slice of the test image using `single_img_features`. In this function, we pass in our feature extraction parameters which we used on our training data, and apply the same feature extraction methods on this single slice of the image. Once we have collected the color_hist, HOG, and spatial features, we return a concatenated array and do a transform using the `Scaler.transform()` function. Finally, we call our classifier and try to predict if this window contains a car or not. If it does, we append it to our `on_windows` list. This process is repeated until we have gone through every window in our list of windows (tiny, small, medium, and large). By the end, we have a list of `on_windows` that our classifier has predicted contains the vehicles detected in the image. These windows contain some vehicles, but as we will soon find out, they also contain a lot of false positives. Below, you can see an example of results from 4 windows where we attempted to detect cars using our search windows:

![alt tag](https://s30.postimg.org/dzcviidlt/final_windows.png)

As you can see, the windows in our frames are able to detect the cars in the images. Wahoo! That means our feature extraction and our windows layout is working. There are a couple of problems with these images though. The first is that these windows cannot possibly be expected to represent car detections, as there are multiple on each car, one that bounds both, and a few others cascaded around them. We need a way to be able to have bounding boxes around just the cars. The other problem is the false positive identified on the divider of the freeway in frames 3 and 4. Our classifier thinks these are vehicles, which they are obviously not. So we need to find a way to filter out this false positive, as well as others like it. Both of these are accomplished with the use of a heatmap.

(window detection for the test images in `test_images` can be found in the `output_images` folder)

## Heatmap
(code for this section can be found in the `HeatMap` class in heat_map.py, starting at line 10)
Ideally, we want to be able to take a frame, identify only the cars in the image, and draw bounding boxes around these cars and nothing else. But with an increased number of search windows that are cascaded in a way that guarantees they will overlap, we are almost always going to end up with multiple windows on our cars, as we saw in the previous image. So how can we draw single images on a car? One way to accomplish this is to use heatmaps from our detections in order to combine overlapping detections. You'll see that the heatmap can also help with removing false positives from our frame, by decreasing their heat and thresholding them out.

### Overlapping Detections
(code for this section can be found in `draw_labeled_boxes` in `P5_video.py`, starting at line 22)
To make a heat-map, the idea is to add "heat" to all the pixels within our classifier's positive-detection windows. We have created a new class called `Heatmap` which takes the windows of a given image and adds heat to each pixels within that window. In the `add_heat` function, we add our heat factor (`self.heat = 0.5`) to each window. We also apply a threshold to the windows. Using the `apply_threshold()` function, we "0 out" any value in the heatmap that is less than two. THere is what the heatmaps of the previous 4 frames looks like:

![alt tag](https://s30.postimg.org/vbd83y735/final_heatmap.png)

The library used to produce this heatmap is Labels in `scipy.ndimage.measurements ` With this heatmap, we call `draw_labeled_bboxes(image, labels)` which draws bounding boxes onto our image around the hot parts that are near each other, which in our case should symbolize the cars. Here is what the final result looks like:

![alt tag](https://s28.postimg.org/44a23ghrh/figure_3.png)

As you can see, the hottest parts of the heat map were bounded with rectangles, which are clearly identifying the cars. You will also notice that the false positives are now gone. This is because in the `add_heat` function of our heatmap, we also use a thresholding step that wipes out any areas in the image that were identified by less than 2 windows. Thus, if our SVC identifies 4 or 5 windows around our vehicles, we continue adding 1 to our heatmap around those windows, but if it only finds one window around a false positive, this one window will be erased when we set our threshold to a minimum of 2 windows.


## From Pipeline to Video
(Code for this section can be found in `P5_video.py`, line 198)

Video can be viewed [here] (https://youtu.be/s4fORr56yJA)

The transition to implementing our pipeline on the project video was a steep one, but in the end we found that utilizing the powers of the heatmap, as well as a few other tricks, allowed us to keep a consistent frame-to-frame transition. Here are a few of the techniques we used in transitioning our pipeline to the video:

### Storing Previous Heatmaps
(code for this section can be found in the `HeatMap` class in heat_map.py, starting at line 10)

In our lane-finding project, we learned about a very useful trick to identify the lane in the following frame. Rather than blindly looking for the frame, we would store the center of the left and right lane from the previous frame of the video, and use that as a starting point for our new frame. For this project, we used that trick in storing windows detected in the previous frame, and adding extra heat to a frame if it was detected in the pervious frame. In our `add_heat` method, we would iterate through the windows from the previous frame while iterating through the new ones, and increment the heatmap by one whenever we came across a window that matched. 

### Chilling and Saturation
To avoid detections being left over for too many frames, we developed a `chill()` method that would reduce every pixel in our heatmap by `self.chill = 4`. This way, it would not be biased towards previously high detected pixels if they were not highly detected in our current frame. It also helped get rid of false positives that may have been identified in both our current and previous frames. We also added a saturation step where by setting `self.saturation = 10`, we would call `                self.heatmap[np.where(self.heatmap > self.saturation)] = self.saturation` so that even with consistently heating up a given window, it would never reach a level where our chilling would not be able to bring it back down. One problem we faced was that when we would be transitioning from one frame to the next, the bounding box on a car would drag on for a few frames and slowly shortten, almost like a tail following the car. We later realized that the problem was that we were not "chilling fast enough". Pixels that may have been hot 4 frames ago were still being identified as hot enough to be detected, so by being caught in our bounding box they would extend it unecessarily wide. By increasing our chill rate to 4, we were able to filter out "lagging" pixels and fixed the issue of our bounding boxes dragging. This also helped filter out some remaining false positives as well, which would be washed out at the beginning of the frame by this increased chill rate.

### Adding More Windows
(code for this section can be found in `P5_video.py`, in the `get_windows()` function, starting at line 137)

We realized that the amount of windows we had been using were not getting enough detections, so in `get_windows`, we added 2 more layers of small windows along the horizon and made sure to make them overlap with each other, as well as the windows that we had previosuly drawn. Althogh adding more windows increased our compile time for the video, since each frame would have to now draw more windows, in the end it helped tremendously with increasing the number of windows detected. 



## Discussion

For our last project of the class, this was a challenging but rewarding adventure. We were given the chance to call upon our knowledge of computer vision and machine learning together, while learning about new ways to extract features from images such as HOG and new classifiers to use. I was especially impressed with SVC which although initially seemed quite basic when compard to the deep neural networks we had designed in the past, was able to classify the images with remarkably high accuracy! I thuroughly enjoyed the process of developing the pipeline and although the transition from running the pipeline from a single image to a video can be an uphill battle, it is very much worth the trouble when you are able to see your work flow across thousands of images. 

The difficulty in this project was extracting features and dealing with the false positives. Hours were spent fiddleing with different parameters in the HOG and `bin_spatial` feature extraction, and although it did help bring to light how each cell in each image is taken into account when determining these features, it was a tedious process. We settled on YUV color channel in the end, as the others we tried with our other parameters we were unable to detect the white car. In other hypotehtical scenarios, our pipeline may not perform as well on other colored cars, and our parameters may have to be revaluated to work seemlessly on all roads and cars.

False positives caused a big problem in our pipeline as well. Even in our final video, you may still see a couple of false positives that appear for a few seconds, but it is considerably improved from before. We found that a chill rate of 4 and a heating rate of 1 was sufficient to not have false positives repeat over a series of frames, as long as our cars had enough detections in the frame to make them reappear. To do this, we added multiple layers of new search windows and made sure to overlap them over existing windows. While this did hurt compile time, it was necessary to improve our ratio of detected cars to non-cars to be able to justify a 1 heating rate. We also collected some of the false positives that were detected in the beginning of the video and added them to the `non-cars` feature list. It may have been that these features, such as the freeway sign or the divider, were not well represented in the data we were given, but as we have learned the easiest way to improve your model is by adding more data to it. This did help in diminishing false positive detection in the beginning of the video, but if our pipeline was to be tried on a different video, new false positives would be expected to appear that we may not have accounted for that would cause false detections in our windows.

Overall, we are really happy with our resulting video and enjoyed working on this project. It was a pleasure starting from basic features, to extracting them from an image, to finally being able to identify cars in a series of frames in a video. Classical computer vision techniques still amaze us to this day, and there is so much that can be done in the world of self-driving cars with these methods available to us.
