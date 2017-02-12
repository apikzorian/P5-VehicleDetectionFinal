import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import cv2
import glob
import time
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from skimage.feature import hog
from lesson_functions import *
# NOTE: the next import is only valid for scikit-learn version <= 0.17
# for scikit-learn >= 0.18 use:
# from sklearn.model_selection import train_test_split
from sklearn.cross_validation import train_test_split
from sklearn.utils import shuffle
from skimage.feature import hog
import pickle
from moviepy.video.io.VideoFileClip import VideoFileClip
from heat_map import *

# Iterates through detected cars and draws bounding boxes over them
def draw_labeled_bboxes(img, labels):
    # Iterate through all detected cars
    draw_img = np.copy(img)
    for car_number in range(1, labels[1]+1):
        # Find pixels with each car_number label value
        nonzero = (labels[0] == car_number).nonzero()
        # Identify x and y values of those pixels
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Define a bounding box based on min/max x and y
        bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
        # Draw the box on the image
        cv2.rectangle(draw_img, bbox[0], bbox[1], (0,0,255), 6)
    # Return the image
    return draw_img

# Define a function to extract features from a single image window
# This function is very similar to extract_features()
# just for a single image rather than list of images
def single_img_features(img, color_space='RGB', spatial_size=(32, 32),
                        hist_bins=32, orient=9,
                        pix_per_cell=8, cell_per_block=2, hog_channel=0,
                        spatial_feat=True, hist_feat=True, hog_feat=True, feature_vec = False):
    # 1) Define an empty list to receive features
    img_features = []
    # 2) Apply color conversion if other than 'RGB'
    if color_space != 'RGB':
        if color_space == 'HSV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        elif color_space == 'LUV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2LUV)
        elif color_space == 'HLS':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
        elif color_space == 'YUV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
        elif color_space == 'YCrCb':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    else:
        feature_image = np.copy(img)
    # 3) Compute spatial features if flag is set
    if spatial_feat == True:
        spatial_features = bin_spatial(feature_image, size=spatial_size)
        # 4) Append features to list
        img_features.append(spatial_features)
    # 5) Compute histogram features if flag is set
    if hist_feat == True:
        hist_features = color_hist(feature_image, nbins=hist_bins)
        # 6) Append features to list
        img_features.append(hist_features)
    # 7) Compute HOG features if flag is set
    if hog_feat == True:
        if hog_channel == 'ALL':
            hog_features = []
            for channel in range(feature_image.shape[2]):
                hog_features.extend(get_hog_features(feature_image[:, :, channel],
                                                     orient, pix_per_cell, cell_per_block,
                                                     vis=False, feature_vec=feature_vec))
        else:
            hog_features = get_hog_features(feature_image[:, :, hog_channel], orient,
                                            pix_per_cell, cell_per_block, vis=False, feature_vec=feature_vec)
        # 8) Append features to list
        img_features.append(hog_features)

    # 9) Return concatenated array of features
    return np.concatenate(img_features)


# Define a function you will pass an image
# and the list of windows to be searched (output of slide_windows())
def search_windows(img, windows, clf, scaler, color_space='RGB',
                   spatial_size=(32, 32), hist_bins=32,
                   hist_range=(0, 256), orient=9,
                   pix_per_cell=8, cell_per_block=2,
                   hog_channel=0, spatial_feat=True,
                   hist_feat=True, hog_feat=True):
    # 1) Create an empty list to receive positive detection windows
    on_windows = []
    pos_count = 0
    # feature_array = hog(img, orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell),
    #                     cells_per_block=(cell_per_block, cell_per_block), visualise=False, feature_vector=False)

    # feature_array = single_img_features(img, color_space=color_space,
    #                                    spatial_size=spatial_size, hist_bins=hist_bins,
    #                                    orient=orient, pix_per_cell=pix_per_cell,
    #                                    cell_per_block=cell_per_block,
    #                                    hog_channel=hog_channel, spatial_feat=spatial_feat,
    #                                    hist_feat=hist_feat, hog_feat=hog_feat, feature_vec=False)

    # 2) Iterate over all windows in the list
    for window in windows:
        # 3) Extract the test window from original image
        test_img = cv2.resize(img[window[0][1]:window[1][1], window[0][0]:window[1][0]], (64, 64))
        # 4) Extract features for that window using single_img_features()
        features = single_img_features(test_img, color_space=color_space,
                                       spatial_size=spatial_size, hist_bins=hist_bins,
                                       orient=orient, pix_per_cell=pix_per_cell,
                                       cell_per_block=cell_per_block,
                                       hog_channel=hog_channel, spatial_feat=spatial_feat,
                                       hist_feat=hist_feat, hog_feat=hog_feat, feature_vec=True)
        # 5) Scale extracted features to be fed to classifier
        test_features = scaler.transform(np.array(features).reshape(1, -1))
        # 6) Predict using your classifier
        prediction = clf.predict(test_features)
        # 7) If positive (prediction == 1) then save the window
        if prediction == 1:
            # global count
            #path = "trimmed_positives/frame_" + str(count) + "_pos_" + str(pos_count) + ".jpg"
            path = "false_positives/frame_" + str(count) + "_pos_" + str(pos_count) + "za.jpg"
            cv2.imwrite(path, test_img)  # save frame as JPEG file
            pos_count += 1
            on_windows.append(window)
    # 8) Return windows for positive detections
    return on_windows

# Return windows of various sizes, which will be used to search image
def get_windows(img):
    tiny_windows = slide_window(img, x_start_stop=[None, None], y_start_stop=[300, 700], xy_window=(96, 96),
                                xy_overlap=(0.5, 0.5))
    tiny_windows2 = slide_window(img, x_start_stop=[400, None], y_start_stop=[280, 700], xy_window=(96, 96),
                                 xy_overlap=(0.5, 0.5))

    tiny_windows3 = slide_window(img, x_start_stop=[550, None], y_start_stop=[290, 700], xy_window=(96, 96),
                                 xy_overlap=(0.5, 0.5))


    small_windows = slide_window(img, x_start_stop=[100, None], y_start_stop=[410, 700], xy_window=(384, 96),
                                 xy_overlap=(0.5, 0.5))

    medium_windows = slide_window(img, x_start_stop=[400, None], y_start_stop=[500, 700], xy_window=(576, 288),
                                  xy_overlap=(0.5, 0.5))

    windows = tiny_windows + tiny_windows2 + tiny_windows3 + small_windows + medium_windows

    return windows

# Pipeline function to be used on each frame in video. Extract features, cascade search windows over image, detect cars
# in image, and draw bounding boxes over the car
def process_frame(image):

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

    windows = get_windows(image)

    hot_windows = search_windows(image, windows, svc, X_scaler, color_space=color_space,
                                 spatial_size=spatial_size, hist_bins=hist_bins,
                                 orient=orient, pix_per_cell=pix_per_cell,
                                 cell_per_block=cell_per_block,
                                 hog_channel=hog_channel, spatial_feat=spatial_feat,
                                 hist_feat=hist_feat, hog_feat=hog_feat)

    heat_windows = heat_map.add_heat(hot_windows)
    labels = label(heat_windows)

     global count
    if count > 500:
        heat_map.set_threshold(2)
    cv2.imwrite("in_frames_0211_project/in_frame%d.jpg" % count, image)  # save frame as JPEG file

    hot_box = draw_labeled_bboxes(image, labels)
    cv2.imwrite("out_frames_0211_project/out_frame%d.jpg" % count, hot_box)  # save frame as JPEG file
    count += 1

    return hot_box


# Trains SVC classifier on training images and labels. Creates testing images and labels, tests
# the classifier, and returns training features Scalar and SVC classifier
def get_classifier():
    with open("P5_features_labels2.p", mode='rb') as f:
        pfile = pickle.load(f)
    X = pfile["features"]
    y = pfile["y"]

    # Fit a per-column scaler
    X_scaler = StandardScaler().fit(X)
    # Apply the scaler to X
    scaled_X = X_scaler.transform(X)

    # Shuffle features and labels
    scaled_X, y = shuffle(scaled_X, y)

    # Split up data into randomized training and test sets
    rand_state = np.random.randint(0, 100)
    X_train, X_test, y_train, y_test = train_test_split(
        scaled_X, y, test_size=0.2, random_state=rand_state)

    print('Feature vector length:', len(X_train[0]))
    # Use a linear SVC
    svc = LinearSVC()
    # Check the training time for the SVC
    t = time.time()
    svc.fit(X_train, y_train)
    t2 = time.time()
    print(round(t2 - t, 2), 'Seconds to train SVC...')
    # Check the score of the SVC
    print('Test Accuracy of SVC = ', round(svc.score(X_test, y_test), 4))
    # Check the prediction time for a single sample
    t = time.time()

    return X_scaler, svc

X_scaler, svc = get_classifier()
count = 0
video_output = 'project_output.mp4'
clip1 = VideoFileClip("project_video.mp4")

start_image = mpimg.imread('in_frames_0211/in_frame53.jpg') #Standard size of image from video to scale heatmap
heat_map = HeatMap(start_image) #Heatmap to be used throughout pipeline

# Perform image processing on each frame and save new video
video_clip = clip1.fl_image(process_frame)
video_clip.write_videofile(video_output, audio=False)