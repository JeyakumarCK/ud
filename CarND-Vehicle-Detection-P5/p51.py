import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import cv2
import glob
import os
import time
import pickle
from moviepy.editor import VideoFileClip
from skimage.feature import hog
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.cross_validation import train_test_split
from scipy.ndimage.measurements import label
from tqdm import tqdm

#-------------------------------------------------------------------#
#Parameters to tweak for feature extraction
#-------------------------------------------------------------------#
color_space = 'YCrCb' # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
orient = 8  # HOG orientations
pix_per_cell = 8 # HOG pixels per cell
cell_per_block = 1 # HOG cells per block
hog_channel = 'ALL' # Can be 0, 1, 2, or "ALL"
spatial_size = (16, 16) # Spatial binning dimensions
hist_bins = 16    # Number of histogram bins
spatial_feat = True # Spatial features on or off
hist_feat = True # Histogram features on or off
hog_feat = True # HOG features on or off
y_start_stop1 = [350, 650] # Min and max in y to search in slide_window()
y_start_stop2 = [350, 650] # Min and max in y to search in slide_window()
y_start_stop3 = [350, 700] # Min and max in y to search in slide_window()
y_start_stop4 = [350, 700] # Min and max in y to search in slide_window()

xy_window1=(80, 80) # window size to use in slide window function 
xy_window2=(96, 96) # window size to use in slide window function 
xy_window3=(128, 128) # window size to use in slide window function 
xy_window4=(160, 160) # window size to use in slide window function 

xy_overlap=(0.50, 0.50)# window overlap in slide window function
svc = None
windows = None
last_hot_boxes = []
no_of_last_boxes = 12


#-------------------------------------------------------------------#
#Function to smoothen the hotboxes detected with previous detections
#-------------------------------------------------------------------#
def smooth_hotboxes(hot_windows):
    global last_hot_boxes
    
    if len(last_hot_boxes) >= no_of_last_boxes:
        last_hot_boxes.pop(0)
    last_hot_boxes.append(hot_windows)

    new_hot_windows = []
    for boxes in last_hot_boxes:
        new_hot_windows.extend(boxes)
    
    return new_hot_windows


'''
# A reasonably good setting that produced almost all findings
color_space = 'YCrCb' # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
orient = 9  # HOG orientations
pix_per_cell = 8 # HOG pixels per cell
cell_per_block = 2 # HOG cells per block
hog_channel = 'ALL' # Can be 0, 1, 2, or "ALL"
spatial_size = (16, 16) # Spatial binning dimensions
hist_bins = 16    # Number of histogram bins
spatial_feat = True # Spatial features on or off
hist_feat = True # Histogram features on or off
hog_feat = True # HOG features on or off
y_start_stop = [350, 700] # Min and max in y to search in slide_window()
xy_window=(96, 96) # window size to use in slide window function 
xy_overlap=(0.5, 0.5) # window overlap in slide window function

'''

#-------------------------------------------------------------------#
#Function to load training images in to an array
#-------------------------------------------------------------------#
def loadTrainingImages(pattern='training_data/**/*.png'):
    images = glob.glob(pattern, recursive=True)
    cars = []
    notcars = []
    
    for image in tqdm(images):
        if 'non-vehicles' in image:
            notcars.append(mpimg.imread(image))
        else:
            cars.append(mpimg.imread(image))
    
    print('count of cars:', len(cars), ', shape of one image is', cars[10].shape, ', and datatype is', cars[10].dtype)
    print('count of notcars:', len(notcars), ', shape of one image is', notcars[11].shape, ', and datatype is', notcars[10].dtype)
    return cars, notcars

#-------------------------------------------------------------------#
#Function to load image file names into an array
#-------------------------------------------------------------------#
def loadTrainingImageFiles(pattern='training_data/**/*.png'):
    images = glob.glob(pattern, recursive=True)
    cars = []
    notcars = []
    
    for image in tqdm(images):
        if 'non-vehicles' in image:
            notcars.append(image)
        else:
            cars.append(image)
    
    print('Total count of cars:', len(cars))
    print('Total count of notcars:', len(notcars))
    return cars, notcars

#-------------------------------------------------------------------#
#Define a function to return HOG features and visualization
#-------------------------------------------------------------------#
def get_hog_features(img, orient, pix_per_cell, cell_per_block, 
                        vis=False, feature_vec=True):
    # Call with two outputs if vis==True
    if vis == True:
        features, hog_image = hog(img, orientations=orient, 
                                  pixels_per_cell=(pix_per_cell, pix_per_cell),
                                  cells_per_block=(cell_per_block, cell_per_block), 
                                  transform_sqrt=True, 
                                  visualise=vis, feature_vector=feature_vec)
        return features, hog_image
    # Otherwise call with one output
    else:      
        features = hog(img, orientations=orient, 
                       pixels_per_cell=(pix_per_cell, pix_per_cell),
                       cells_per_block=(cell_per_block, cell_per_block), 
                       transform_sqrt=True, 
                       visualise=vis, feature_vector=feature_vec)
        return features

#-------------------------------------------------------------------#
# Define a function to compute binned color features  
#-------------------------------------------------------------------#
def bin_spatial(img, size=(32, 32)):
    # Use cv2.resize().ravel() to create the feature vector
    features = cv2.resize(img, size).ravel() 
    # Return the feature vector
    return features

#-------------------------------------------------------------------#
# Define a function to compute color histogram features 
# NEED TO CHANGE bins_range to (0, 1) if reading .png files with mpimg! else (0, 255)
#-------------------------------------------------------------------#
def color_hist(img, nbins=32, bins_range=(0, 1)):
    # Compute the histogram of the color channels separately
    channel1_hist = np.histogram(img[:,:,0], bins=nbins)
    channel2_hist = np.histogram(img[:,:,1], bins=nbins)
    channel3_hist = np.histogram(img[:,:,2], bins=nbins)
    # Concatenate the histograms into a single feature vector
    hist_features = np.concatenate((channel1_hist[0], channel2_hist[0], channel3_hist[0]))
    # Return the individual histograms, bin_centers and feature vector
    return hist_features

#-------------------------------------------------------------------#
# Define a function to extract features from a list of images
# Have this function call bin_spatial() and color_hist()
#-------------------------------------------------------------------#
def extract_features(imgs, color_space='RGB', spatial_size=(32, 32),
                        hist_bins=32, orient=9, 
                        pix_per_cell=8, cell_per_block=2, hog_channel=0,
                        spatial_feat=True, hist_feat=True, hog_feat=True):
    # Create a list to append feature vectors to
    features = []
    # Iterate through the list of images
    for file in imgs:
        #file_features = []
        # Read in each one by one
        image = mpimg.imread(file)
        file_features = single_img_features(image, color_space=color_space, 
                            spatial_size=spatial_size, hist_bins=hist_bins, 
                            orient=orient, pix_per_cell=pix_per_cell, 
                            cell_per_block=cell_per_block, 
                            hog_channel=hog_channel, spatial_feat=spatial_feat, 
                            hist_feat=hist_feat, hog_feat=hog_feat)

        features.append(file_features)
    # Return list of feature vectors
    return features
    
#-------------------------------------------------------------------#
# Define a function that takes an image,
# start and stop positions in both x and y, 
# window size (x and y dimensions),  
# and overlap fraction (for both x and y)
#-------------------------------------------------------------------#
def slide_window(img, x_start_stop=[None, None], y_start_stop=[None, None], 
                    xy_window=(64, 64), xy_overlap=(0.5, 0.5)):
    # If x and/or y start/stop positions not defined, set to image size
    if x_start_stop[0] == None:
        x_start_stop[0] = 0
    if x_start_stop[1] == None:
        x_start_stop[1] = img.shape[1]
    if y_start_stop[0] == None:
        y_start_stop[0] = 0
    if y_start_stop[1] == None:
        y_start_stop[1] = img.shape[0]
    # Compute the span of the region to be searched    
    xspan = x_start_stop[1] - x_start_stop[0]
    yspan = y_start_stop[1] - y_start_stop[0]
    # Compute the number of pixels per step in x/y
    nx_pix_per_step = np.int(xy_window[0]*(1 - xy_overlap[0]))
    ny_pix_per_step = np.int(xy_window[1]*(1 - xy_overlap[1]))
    # Compute the number of windows in x/y
    nx_buffer = np.int(xy_window[0]*(xy_overlap[0]))
    ny_buffer = np.int(xy_window[1]*(xy_overlap[1]))
    nx_windows = np.int((xspan-nx_buffer)/nx_pix_per_step) 
    ny_windows = np.int((yspan-ny_buffer)/ny_pix_per_step) 
    # Initialize a list to append window positions to
    window_list = []
    # Loop through finding x and y window positions
    # Note: you could vectorize this step, but in practice
    # you'll be considering windows one by one with your
    # classifier, so looping makes sense
    for ys in range(ny_windows):
        for xs in range(nx_windows):
            # Calculate window position
            startx = xs*nx_pix_per_step + x_start_stop[0]
            endx = startx + xy_window[0]
            starty = ys*ny_pix_per_step + y_start_stop[0]
            endy = starty + xy_window[1]
            
            # Append window position to list
            window_list.append(((startx, starty), (endx, endy)))
    # Return the list of windows
    return window_list

#-------------------------------------------------------------------#
# Define a function to draw bounding boxes
#-------------------------------------------------------------------#
def draw_boxes(img, bboxes, color=(0, 0, 255), thick=6):
    # Make a copy of the image
    imcopy = np.copy(img)
    # Iterate through the bounding boxes
    for bbox in bboxes:
        # Draw a rectangle given bbox coordinates
        cv2.rectangle(imcopy, bbox[0], bbox[1], color, thick)
    # Return the image copy with boxes drawn
    return imcopy

#-------------------------------------------------------------------#
# Define a function to extract features from a single image window
# This function is very similar to extract_features()
# just for a single image rather than list of images
#-------------------------------------------------------------------#
def single_img_features(img, color_space='RGB', spatial_size=(32, 32),
                        hist_bins=32, orient=9, 
                        pix_per_cell=8, cell_per_block=2, hog_channel=0,
                        spatial_feat=True, hist_feat=True, hog_feat=True):    
    #1) Define an empty list to receive features
    img_features = []
    #2) Apply color conversion if other than 'RGB'
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
    else: feature_image = np.copy(img)      
    #3) Compute spatial features if flag is set
    if spatial_feat == True:
        spatial_features = bin_spatial(feature_image, size=spatial_size)
        #4) Append features to list
        img_features.append(spatial_features)
    #5) Compute histogram features if flag is set
    if hist_feat == True:
        hist_features = color_hist(feature_image, nbins=hist_bins)
        #6) Append features to list
        img_features.append(hist_features)
    #7) Compute HOG features if flag is set
    if hog_feat == True:
        if hog_channel == 'ALL':
            hog_features = []
            for channel in range(feature_image.shape[2]):
                hog_features.extend(get_hog_features(feature_image[:,:,channel], 
                                    orient, pix_per_cell, cell_per_block, 
                                    vis=False, feature_vec=True))      
        else:
            hog_features = get_hog_features(feature_image[:,:,hog_channel], orient, 
                        pix_per_cell, cell_per_block, vis=False, feature_vec=True)
        #8) Append features to list
        img_features.append(hog_features)

    #9) Return concatenated array of features
    return np.concatenate(img_features)

#-------------------------------------------------------------------#
# Define a function you will pass an image 
# and the list of windows to be searched (output of slide_windows())
#-------------------------------------------------------------------#
def search_windows(img, windows, clf, scaler, color_space='RGB', 
                    spatial_size=(32, 32), hist_bins=32, 
                    hist_range=(0, 256), orient=9, 
                    pix_per_cell=8, cell_per_block=2, 
                    hog_channel=0, spatial_feat=True, 
                    hist_feat=True, hog_feat=True):

    #1) Create an empty list to receive positive detection windows
    on_windows = []
    #2) Iterate over all windows in the list
    for window in windows:
        #3) Extract the test window from original image
        test_img = cv2.resize(img[window[0][1]:window[1][1], window[0][0]:window[1][0]], (64, 64))      
        #4) Extract features for that window using single_img_features()
        features = single_img_features(test_img, color_space=color_space, 
                            spatial_size=spatial_size, hist_bins=hist_bins, 
                            orient=orient, pix_per_cell=pix_per_cell, 
                            cell_per_block=cell_per_block, 
                            hog_channel=hog_channel, spatial_feat=spatial_feat, 
                            hist_feat=hist_feat, hog_feat=hog_feat)
        #5) Scale extracted features to be fed to classifier
#        print('scaler', scaler)
#        print('features', features)
        test_features = scaler.transform(np.array(features).reshape(1, -1))
        #6) Predict using your classifier
        prediction = clf.predict(test_features)
        #7) If positive (prediction == 1) then save the window
        if prediction == 1:
            on_windows.append(window)
    #8) Return windows for positive detections
    return on_windows

#-------------------------------------------------------------------#
# Function to normalize the extracted features
#-------------------------------------------------------------------#
def normalize(car_features, notcar_features):
    X = np.vstack((car_features, notcar_features)).astype(np.float64)                        
    # Fit a per-column scaler
    X_scaler = StandardScaler().fit(X)
    # Apply the scaler to X
    scaled_X = X_scaler.transform(X)
    return X_scaler, scaled_X 

#-------------------------------------------------------------------#
# Utility Function to visualize images
#-------------------------------------------------------------------#
def visualize3Images(img1, img2, img3, tit1, tit2, tit3, cmap1, cmap2, cmap3, isImg=False):
    f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(24, 9))
    f.tight_layout()
    ax1.grid(True)
    if isImg:
        ax1.imshow(img1, cmap=cmap1)
    else:
        ax1.imshow(mpimg.imread(img1), cmap=cmap1)
    ax1.set_title(tit1, fontsize=20)
    ax2.grid(True)
    if isImg:
        ax2.imshow(img2, cmap=cmap2)
    else:
        ax2.imshow(mpimg.imread(img2), cmap=cmap2)
    ax2.set_title(tit2, fontsize=20)
    ax3.grid(True)
    if isImg:
        ax3.imshow(img3, cmap=cmap3)
    else:
        ax3.imshow(mpimg.imread(img3), cmap=cmap3)
    ax3.set_title(tit3, fontsize=20)
    plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
    plt.show()


#-------------------------------------------------------------------#
# Utility Function to visualize images
#-------------------------------------------------------------------#
def visualize2Images(img1, img1Tit, img2, img2Tit):
    fig = plt.figure()
    plt.subplot(121)
    plt.imshow(img1)
    plt.title(img1Tit)
    plt.subplot(122)
#    plt.imshow(img2)
#    plt.title(img2Tit)
    plt.imshow(img2, cmap='hot')
    plt.title(img2Tit)
    fig.tight_layout()
    plt.show()

#-------------------------------------------------------------------#
# Function to prepare the training and test dataset
#-------------------------------------------------------------------#
def prepareTrainingAndTestData(sample_size=None):
    cars, notcars = loadTrainingImageFiles()
    if sample_size is not None:
        cars = cars[0:sample_size]
        notcars = notcars[0:sample_size]
    print('cars count to use:', len(cars))
    print('notcars count to use:', len(notcars))
    
    ### TODO: Tweak these parameters and see how the results change.
#    color_space = 'YCrCb' # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
#    orient = 9  # HOG orientations
#    pix_per_cell = 8 # HOG pixels per cell
#    cell_per_block = 2 # HOG cells per block
#    hog_channel = 0 # Can be 0, 1, 2, or "ALL"
#    spatial_size = (32, 32) # Spatial binning dimensions
#    hist_bins = 16    # Number of histogram bins
#    spatial_feat = True # Spatial features on or off
#    hist_feat = True # Histogram features on or off
#    hog_feat = True # HOG features on or off
#    y_start_stop = [350, 700] # Min and max in y to search in slide_window()
    
    car_features = extract_features(cars, color_space=color_space, 
                            spatial_size=spatial_size, hist_bins=hist_bins, 
                            orient=orient, pix_per_cell=pix_per_cell, 
                            cell_per_block=cell_per_block, 
                            hog_channel=hog_channel, spatial_feat=spatial_feat, 
                            hist_feat=hist_feat, hog_feat=hog_feat)
    notcar_features = extract_features(notcars, color_space=color_space, 
                            spatial_size=spatial_size, hist_bins=hist_bins, 
                            orient=orient, pix_per_cell=pix_per_cell, 
                            cell_per_block=cell_per_block, 
                            hog_channel=hog_channel, spatial_feat=spatial_feat, 
                            hist_feat=hist_feat, hog_feat=hog_feat)

    print('len(car_features)', len(car_features), len(car_features[10]))
    print('len(notcar_features)', len(notcar_features), len(notcar_features[11]))
    
    X_scaler, scaled_X = normalize(car_features, notcar_features)

    # Define the labels vector
    y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))

    # Split up data into randomized training and test sets
    rand_state = np.random.randint(0, 100)
    X_train, X_test, y_train, y_test = train_test_split(scaled_X, y, test_size=0.2, random_state=rand_state)
    return X_scaler, X_train, X_test, y_train, y_test

#-------------------------------------------------------------------#
# Function that gives us a trained model to use
#-------------------------------------------------------------------#
def getTrainedModel(pf=None):
    # If pickle file having svc exists, read that file and return svc
    # else read the test data and train the model & return it.
    if pf is not None and os.path.isfile(pf):
        print('trained svc model is read from pickle file', pf)
        with open(pf, 'rb') as AutoPickleFile:
             X_scaler, svc = pickle.load(AutoPickleFile)    
    else:
        print('svc model is trained from scratch')
        X_scaler, X_train, X_test, y_train, y_test = prepareTrainingAndTestData()
        # Use a linear SVC 
        svc = LinearSVC()
        # Check the training time for the SVC
        t=time.time()
        svc.fit(X_train, y_train)
        t2 = time.time()
        print(round(t2-t, 2), 'Seconds to train SVC...')
        # Check the score of the SVC
        print('Test Accuracy of SVC = ', round(svc.score(X_test, y_test), 4))
        # Check the prediction time for a single sample
        t=time.time()
        
        if pf is not None:
            with open(pf, 'wb') as AutoPickleFile:
                pickle.dump((X_scaler, svc), AutoPickleFile)
            print('trained svc model written to pickle file', pf)
    
    return X_scaler, svc

#-------------------------------------------------------------------#
# Utility Function to visualize HOG images
#-------------------------------------------------------------------#
def visualizeHOG(img):
    img_original = np.copy(img)
    
    color_space = 'YCrCb' # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
    orient = 12  # HOG orientations
    pix_per_cell = 4 # HOG pixels per cell
    cell_per_block = 1 # HOG cells per block
    hog_channel = 1 # Can be 0, 1, 2, or "ALL"
    
    img = cv2.cvtColor(img, cv2.COLOR_RGB2YCR_CB)
    features, hog_image = get_hog_features(img[:,:,2], orient, pix_per_cell, cell_per_block, vis=True, feature_vec=False)
    #visualize3Images(img_original, img[:,:,1], hog_image, 'HOG Visualization of V Channel of HSV', True)
    visualize3Images(img_original, img[:,:,2], hog_image, 
                     'Original Image', 'Image Channel', color_space, None, 'gray', 'gray', isImg=True)

#-------------------------------------------------------------------#
# Function to add heat to the heatmap
#-------------------------------------------------------------------#
def add_heat(heatmap, bbox_list):
    # Iterate through list of bboxes
    for box in bbox_list:
        # Add += 1 for all pixels inside each bbox
        # Assuming each "box" takes the form ((x1, y1), (x2, y2))
        heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1

    # Return updated heatmap
    return heatmap# Iterate through list of bboxes
    
#-------------------------------------------------------------------#
# Function to apply threshold on the detection to remove false positives
#-------------------------------------------------------------------#
def apply_threshold(heatmap, threshold):
    # Zero out pixels below the threshold
    heatmap[heatmap <= threshold] = 0
    # Return thresholded map
    return heatmap

#-------------------------------------------------------------------#
# Function that draws rectangles on the images with the given labels
#-------------------------------------------------------------------#
def draw_labeled_bboxes(img, labels):
    # Iterate through all detected cars
    for car_number in range(1, labels[1]+1):
        # Find pixels with each car_number label value
        nonzero = (labels[0] == car_number).nonzero()
        # Identify x and y values of those pixels
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Define a bounding box based on min/max x and y
        bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
        # Draw the box on the image
        cv2.rectangle(img, bbox[0], bbox[1], (0,0,255), 6)
    # Return the image
    return img

#-------------------------------------------------------------------#
# Function that processes a given image
#-------------------------------------------------------------------#
def process_image(image):
    draw_image = np.copy(image)
    # Uncomment the following line if you extracted training
    # data from .png images (scaled 0 to 1 by mpimg) and the
    # image you are searching is a .jpg (scaled 0 to 255)
    image = image.astype(np.float32)/255
    global windows
    if windows is None:
        print('inside windows loop')
        windows = slide_window(image, x_start_stop=[None, None], y_start_stop=y_start_stop1, 
                            xy_window=xy_window1, xy_overlap=xy_overlap)
        #temp_img1 = draw_boxes(np.copy(draw_image), windows, color=(255, 0, 0), thick=4)
        #print('I-windows to search', len(windows))
        windows += slide_window(image, x_start_stop=[None, None], y_start_stop=y_start_stop2, 
                            xy_window=xy_window2, xy_overlap=xy_overlap)
        #temp_img2 = draw_boxes(np.copy(draw_image), windows, color=(0, 255, 0), thick=4)
        #print('II-windows to search', len(windows))
        windows += slide_window(image, x_start_stop=[None, None], y_start_stop=y_start_stop3, 
                            xy_window=xy_window3, xy_overlap=xy_overlap)
        #temp_img3 = draw_boxes(np.copy(draw_image), windows, color=(0, 0, 255), thick=4)
        #print('total windows to search', len(windows))
    
    print('windows size', len(windows))
    hot_windows = search_windows(image, windows, clf, X_scaler, color_space=color_space, 
                            spatial_size=spatial_size, hist_bins=hist_bins, 
                            orient=orient, pix_per_cell=pix_per_cell, 
                            cell_per_block=cell_per_block, 
                            hog_channel=hog_channel, spatial_feat=spatial_feat, 
                            hist_feat=hist_feat, hog_feat=hog_feat)                       
    
    print('hot_windows', len(hot_windows))
    hot_windows = smooth_hotboxes(hot_windows)
    print('new hot_windows', len(hot_windows))
    #print('hot_windows', hot_windows)
#    window_img = draw_boxes(np.copy(draw_image), hot_windows, color=(0, 255, 0), thick=5)
    heat = np.zeros_like(image[:,:,0]).astype(np.float)
    heat = add_heat(heat, hot_windows)
    heat = apply_threshold(heat,2)
    heatmap = np.clip(heat, 0, 255)
    labels = label(heatmap)
    final_img = draw_labeled_bboxes(np.copy(draw_image), labels)
#    visualize3Images(draw_image, window_img, final_img, 
#                     'Original', 'Raw Detection', 'Average Box', 
#                     None, None, None, True)
    return labels, final_img
    return final_img

#-------------------------------------------------------------------#
# Function to process images in a given folder/file pattern
#-------------------------------------------------------------------#
def process_images(fn_pattern):
    imgs = glob.glob(fn_pattern)
    for imgf in imgs:
        print('img file', imgf)
        image = mpimg.imread(imgf)
        labels, final_img = process_image(image)

#-------------------------------------------------------------------#
# Function to process video
#-------------------------------------------------------------------#
def process_video(vfn):
    out_vfn = 'out_' + vfn
    vclip = VideoFileClip(vfn)
    #vclip = vclip.subclip(20, 30)
    processed_vclip = vclip.fl_image(process_image)
    processed_vclip.write_videofile(out_vfn, audio=False)
    print('---------------Video Processing Completed------------')
    

#-------------------------------------------------------------------#
# main function starts here
#-------------------------------------------------------------------#
X_scaler, clf = getTrainedModel('scaler_svc.p')
#process_images('test_images/*.jpg')
process_video('p4_project_video.mp4')
#process_video('test_video.mp4')


#print('X_train.shape', X_train.shape, 'len(y_train)', len(y_train))
#print('X_test.shape', X_test.shape, 'len(y_test)', len(y_test))
#cars, notcars = loadTrainingImages('training_data/**/*1.png')
#c_i = np.random.randint(0, len(cars)-1, size=3)
#nc_i = np.random.randint(0, len(notcars)-1, size=3)
#visualize3Images(cars[c_i[0]], cars[c_i[1]], cars[c_i[2]], 'Random Sample from Cars', 'Random Sample from Cars', 'Random Sample from Cars', None, None, None, isImg=True)
#visualize3Images(notcars[nc_i[0]], notcars[nc_i[1]], notcars[nc_i[2]], 'Random Sample from NotCars', 'Random Sample from NotCars', 'Random Sample from NotCars', None, None, None, isImg=True)
#visualizeHOG(cars[c_i[0]])
#visualizeHOG(notcars[nc_i[0]])
