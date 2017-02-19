import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob
import os
import pickle
from moviepy.editor import VideoFileClip
from tqdm import tqdm

#-----------------------------------------------------------------------#
# Define a class to receive the characteristics of each line detection
#-----------------------------------------------------------------------#
class Line():
    def __init__(self):
        self.detected = False                   # was the line detected in the last iteration?
        self.recent_xfitted = []                # x values of the last n fits of the line
        self.bestx = None                       #average x values of the fitted line over the last n iterations
        self.best_fit = None                    #polynomial coefficients averaged over the last n iterations
        self.current_fit = [np.array([False])]  #polynomial coefficients for the most recent fit
        self.radius_of_curvature = None         #radius of curvature of the line in some units
        self.line_base_pos = None               #distance in meters of vehicle center from the line
        self.diffs = np.array([0,0,0], dtype='float') #difference in fit coefficients between last and new fits
        self.allx = None                        #x values for detected line pixels
        self.ally = None                        #y values for detected line pixels

#-----------------------------------------------------------------------#
# Calibrate Camera using Sample Images provided and store it in a Pickle
# file.  If there is a pickle file exists already, then read that file
# and return the Camera Matrix (mtx) and Distortion Coefficients (dist)
#-----------------------------------------------------------------------#
def calibrateCameraBySamples(samples_folder, file_name_pattern, nx, ny, pickle_file_name='cam_calib_mtx_dist.p', save_calibration=True):
    mtx, dist = None, None
    pickle_file_path_name = None
    if (pickle_file_name is not None):
        pickle_file_path_name = os.path.join(samples_folder, pickle_file_name)
    
    if os.path.isfile(pickle_file_path_name):
        # Read in the saved camera matrix and distortion coefficients
        print('Reading calibration from pickle file', pickle_file_path_name)
        dist_pickle = pickle.load( open( pickle_file_path_name, "rb" ) )
        mtx = dist_pickle["mtx"]
        dist = dist_pickle["dist"]
    else:
        print('Generating calibration from sample images in ', samples_folder)
        objp = np.zeros((nx * ny, 3), np.float32)
        objp[:, :2] = np.mgrid[0:nx, 0:ny].T.reshape(-1, 2)
    
        objpoints = []
        imgpoints = []
        img_size = None
        
        fnames = os.path.join(samples_folder, file_name_pattern)
        print(fnames)
        images = glob.glob(fnames)
        for idx, fname in tqdm(enumerate(images)):
            img = mpimg.imread(fname)
            if img_size == None:
                img_size = img.shape[0:2]
                print('img_size = ', img_size)
            ret, corners = cv2.findChessboardCorners(img, (nx,ny))
            #print(idx, fname, ret)
            if ret == True:
                objpoints.append(objp)
                imgpoints.append(corners)
        
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size, None, None)
        if save_calibration == True:
            dist_pickle = {}
            dist_pickle['mtx'] = mtx
            dist_pickle['dist'] = dist
            print('Storing mtx and dist values in', pickle_file_path_name)
            pickle.dump( dist_pickle, open( pickle_file_path_name, "wb" ) )
    
    return mtx, dist

#-----------------------------------------------------------------------#
# Verify the Camera Matrix (mtx) and Distortion Coefficients (dist) by
# Undistorting a sample image. Compare the original and undistorted
# images by plotting them side by side
#-----------------------------------------------------------------------#
def testUndistor(img_file_name, mtx, dist):
    img = mpimg.imread(img_file_name)
    dst = cv2.undistort(img, mtx, dist, None, mtx)
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
    f.tight_layout()
    ax1.grid(True)
    ax1.imshow(img)
    ax1.set_title('Original Image', fontsize=20)
    ax2.grid(True)
    ax2.imshow(dst)
    ax2.set_title('UnDistorted Image', fontsize=20)
    plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
    plt.show()        

#-----------------------------------------------------------------------#
# Function that gives out the src and dst coordinates array to use for 
# warp transformations.
#-----------------------------------------------------------------------#
def getTransformationPoints(img):
    s_tl, d_tl = [500, 510], [227, 333]     #top left
    s_tr, d_tr = [790, 510], [1067, 333]    #top right
    s_br, d_br = [1012, 647], [1067, 646]   #bottom left
    s_bl, d_bl = [306, 647], [227, 646]     #bottom right

    src = np.float32([ s_tl, s_tr, s_br, s_bl ])
    dst = np.float32([ d_tl, d_tr, d_br, d_bl ])
#    print('src', src)
#    print('dst', dst)
    return src, dst

#-----------------------------------------------------------------------#
# A function that provides the Transformation matrix & Inverse 
# Transformation for the given src & dst coordinates
#-----------------------------------------------------------------------#
def getTransformationMatrices(src, dst):
    M = cv2.getPerspectiveTransform(src, dst)
    Minv = cv2.getPerspectiveTransform(dst, src)
    return M, Minv

#-----------------------------------------------------------------------#
# A convenience function to view 3 images in a row
#-----------------------------------------------------------------------#
def visualize3Images(img1, img2, img3):
    f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(24, 9))
    f.tight_layout()
    ax1.grid(True)
    ax1.imshow(img1)
    ax1.set_title('Original Image', fontsize=20)
    ax2.grid(True)
    ax2.imshow(img2)
    ax2.set_title('Warped Image', fontsize=20)
    ax3.grid(True)
    ax3.imshow(img3, cmap='gray')
    ax3.set_title('Thresholded Binary Image', fontsize=20)
    plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
    plt.show()        

#-----------------------------------------------------------------------#
# A convenience function to view 2 images in a row
#-----------------------------------------------------------------------#
def visualize2Images(img1, img2):
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
    f.tight_layout()
    ax1.grid(True)
    ax1.imshow(img1)
    ax1.set_title('Oringal Image', fontsize=20)
    ax2.grid(True)
    ax2.imshow(img2, cmap='gray')
    ax2.set_title('Thresholded Color Binary', fontsize=20)
    plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
    plt.show()

#-----------------------------------------------------------------------#
# A convenience function to view one images & a graph plot in a row
#-----------------------------------------------------------------------#
def visualizeImageGraph(img1, grph):
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
    f.tight_layout()
    ax1.grid(True)
    ax1.imshow(img1, cmap='gray')
    ax1.set_title('Warped Threshold Binary', fontsize=20)
    ax2.grid(True)
    ax2.plot(grph)
    ax2.set_title('Histogram', fontsize=20)
    plt.xlim(0, 1280)
    plt.ylim(0, 250)
    plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
    plt.show()        

#-----------------------------------------------------------------------#
# Function to apply Sobel operator for the given orientation & kernal
# and to threshold Sobel derivatives for the given range
#-----------------------------------------------------------------------#
def abs_sobel_thresh(img, orient='x', sobel_kernel=3, thresh=(0, 255)):
    gray = img
    if (len(img.shape) == 3 and img.shape[2] > 2):
        print('abs_sobel_thresh - converting to gray')
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, (1 if orient == 'x' else 0), (1 if orient == 'y' else 0))
    abs_sobelx = np.absolute(sobelx)
    scaled_sobel = np.uint8(255*abs_sobelx/np.max(abs_sobelx))
    binary_output = np.zeros_like(scaled_sobel)
    binary_output[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1
    return binary_output

#-----------------------------------------------------------------------#
# Function to arrive at the Magniture threshold for the given image & 
# threshold range
#-----------------------------------------------------------------------#
def mag_thresh(img, sobel_kernel=3, mag_thresh=(0, 255)):
    gray = img
    if (len(img.shape) == 3 and img.shape[2] > 2):
        print('mag_thresh - converting to gray')
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1,0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0,1, ksize=sobel_kernel)
    abs_sobelxy = np.sqrt(np.square(sobelx) + np.square(sobely))
    scaled_sobelxy = np.uint8( 255*abs_sobelxy/np.max(abs_sobelxy) )
    binary_output = np.zeros_like(scaled_sobelxy)
    binary_output[(scaled_sobelxy >= mag_thresh[0]) & (scaled_sobelxy<= mag_thresh[1])] = 1
    return binary_output
    
#-----------------------------------------------------------------------#
# Function to arrive at the Directional threshold for the given image & 
# threshold range
#-----------------------------------------------------------------------#
def dir_threshold(img, sobel_kernel=3, dir_thresh=(0, np.pi/2)):
    gray = img
    if (len(img.shape) == 3 and img.shape[2] > 2):
        print('dir_threshold - converting to gray')
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1,0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0,1, ksize=sobel_kernel)
    abs_sobelx = np.absolute(sobelx)
    abs_sobely = np.absolute(sobely)
    #abs_sobelxy = np.sqrt(sobelx**2 + sobely**2)
    arc_sobel = np.arctan2(abs_sobely, abs_sobelx)
    binary_output = np.zeros_like(arc_sobel)
    binary_output[(arc_sobel>=dir_thresh[0]) & (arc_sobel<=dir_thresh[1])] = 1
    #print(arc_sobel[200:420, 0:200])
    return binary_output

#-----------------------------------------------------------------------#
# Function to verify the current radius of curvature with the previous 
# frame's Radius of curvature and return the suitable polyfit values
#-----------------------------------------------------------------------#
def lineCheck(line, curverad, fitx, fit):
    # line check for the lane
    if line.detected: # If lane is detected
        # If sanity check passes
        if abs(curverad / line.radius_of_curvature - 1) < .6:        
            print('line.detected is true and roc ratio is < 0.6')    
            line.detected = True
            line.current_fit = fit
            line.allx = fitx
            line.bestx = np.mean(fitx)            
            line.radius_of_curvature = curverad
        # If sanity check fails use the previous values
        else:
            print('line.detected is true and roc ratio is >> 0.6')    
            line.detected = False
            fitx = line.allx
    else:
        # If lane was not detected and no curvature is defined
        if line.radius_of_curvature: 
            if abs(curverad / line.radius_of_curvature - 1) < 1:            
                print('line.detected is false and roc ration is < 1')
                line.detected = True
                line.current_fit = fit
                line.allx = fitx
                line.bestx = np.mean(fitx)            
                line.radius_of_curvature = curverad
            else:
                print('line.detected is false and roc ration is >> 1')
                line.detected = False
                fitx = line.allx      
        # If curvature was defined
        else:
            print('first time data getting set in Line object')
            line.detected = True
            line.current_fit = fit
            line.allx = fitx
            line.bestx = np.mean(fitx)
            line.radius_of_curvature = curverad
    return fitx


#-----------------------------------------------------------------------#
# Function to find the lane line pixels for the given binary warped 
# image and using the line positions of previous frame
#-----------------------------------------------------------------------#
def findXY_NonHistogram(binary_warped, left_fit, right_fit):
    print('Finding fit by non-histogram function')
    # Assume you now have a new warped binary image 
    # from the next frame of video (also called "binary_warped")
    # It's now much easier to find line pixels!
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    margin = 100
    left_lane_inds = ((nonzerox > (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + left_fit[2] - margin)) & (nonzerox < (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + left_fit[2] + margin))) 
    right_lane_inds = ((nonzerox > (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + right_fit[2] - margin)) & (nonzerox < (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + right_fit[2] + margin)))  
    
    # Again, extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]
    
    return leftx, lefty, rightx, righty


#-----------------------------------------------------------------------#
# Function to apply polyfit on the x,y positions and to arrive at the 
# second degree polynomial
#-----------------------------------------------------------------------#
def pixelPositionToXYValues(leftx, lefty, rightx, righty, yvalue):
    # Fit a second order polynomial to each
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)
    
    # Generate x and y values for plotting
    ploty = np.linspace(0, yvalue-1, yvalue)
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2] 
    
    return left_fit, right_fit, left_fitx, right_fitx, ploty
    

#-----------------------------------------------------------------------#
# Function to find the lane line pixels for the given binary warped 
# using the Histogram and Sliding window method
#-----------------------------------------------------------------------#
def findXY_Histogram(binary_warped):
    #1
    histogram = np.sum(binary_warped[binary_warped.shape[0]/2:,:], axis=0)
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
    #visualizeImageGraph(out_img, histogram)

    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0]/2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint
    
    # Choose the number of sliding windows
    nwindows = 9
    # Set height of windows
    window_height = np.int(binary_warped.shape[0]/nwindows)
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    
    # Current positions to be updated for each window
    leftx_current = leftx_base
    rightx_current = rightx_base
    # Set the width of the windows +/- margin
    margin = 100
    # Set minimum number of pixels found to recenter window
    minpix = 50
    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []
    
    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = binary_warped.shape[0] - (window+1)*window_height
        win_y_high = binary_warped.shape[0] - window*window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin
        
        # Draw the windows on the visualization image
        cv2.rectangle(out_img,(win_xleft_low,win_y_low),(win_xleft_high,win_y_high),(0,255,0), 2) 
        cv2.rectangle(out_img,(win_xright_low,win_y_low),(win_xright_high,win_y_high),(0,255,0), 2)
        
        # Identify the nonzero pixels in x and y within the window
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]        
        #print('good_left_inds', good_left_inds)
        #print('good_right_inds', good_right_inds)
        
        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:        
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))
    
    # Concatenate the arrays of indices
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)
    
    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds] 
    
    #For Display 
#    left_fit, right_fit, left_fitx, right_fitx, ploty = pixelPositionToXYValues(leftx, lefty, rightx, righty, out_img.shape[0])
#    out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
#    out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]
#    plt.imshow(binary_warped, cmap='gray')
#    plt.plot(left_fitx, ploty, color='red', linewidth=3.0)
#    plt.plot(right_fitx, ploty, color='red', linewidth=3.0)
#    plt.xlim(0, 1280)
#    plt.ylim(720, 0)
#    plt.show()
    
    return leftx, lefty, rightx, righty #, out_img

#-----------------------------------------------------------------------#
# Function to calculate the radius of the curvature for the given x, y
# values and convert them from pixel to real world metres perspective
#-----------------------------------------------------------------------#
def calculateRadiusOfCurvature(x, y):
    # Define conversions in x and y from pixels space to meters
    ym_per_pix = 30/720 # meters per pixel in y dimension
    xm_per_pix = 3.7/700 # meters per pixel in x dimension
    y_eval = np.max(y)
    
    # Fit new polynomials to x,y in world space
    fit_cr = np.polyfit(y*ym_per_pix, x*xm_per_pix, 2)
    # Calculate the new radii of curvature
    curve_rad = ((1 + (2*fit_cr[0]*y_eval*ym_per_pix + fit_cr[1])**2)**1.5) / np.absolute(2*fit_cr[0])
    # Now our radius of curvature is in meters
    #print(left_curverad, 'm', right_curverad, 'm')
    # Example values: 632.1 m    626.2 m
    return curve_rad

#-----------------------------------------------------------------------#
# Function to calculate the vehicle position with respect to lane lines
# It is calculated based on the image width and lane line coordinates
#-----------------------------------------------------------------------#
def calculateVehiclePosition(image_width, pts):
    # Find the position of the car from the center
    # It will show if the car is 'x' meters from the left or right
    position = image_width/2
    left  = np.min(pts[(pts[:,1] < position) & (pts[:,0] > 700)][:,1])
    right = np.max(pts[(pts[:,1] > position) & (pts[:,0] > 700)][:,1])
    center = (left + right)/2
    # Define conversions in x and y from pixels space to meters
    xm_per_pix = 3.7/700 # meteres per pixel in x dimension   
    #print(position, center, 'in pixels', (position - center))
    return (position - center)*xm_per_pix

#-----------------------------------------------------------------------#
# A convenience Function that draws polygon on the given image, reverts 
# the perspective transformation using Minv and writes the texts.
#-----------------------------------------------------------------------#
def drawPolygonAndUnwrap(binary_warped, undist, pts, Minv, img_size, text1, text2=None):
    # Create an image to draw the lines on
    warp_zero = np.zeros_like(binary_warped).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))
    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))
    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = cv2.warpPerspective(color_warp, Minv, img_size) 
    # Combine the result with the original image
    result = cv2.addWeighted(undist, 1, newwarp, 0.3, 0)

    # Find the position of the car
    pts = np.argwhere(newwarp[:,:,1])
    position = calculateVehiclePosition(undist.shape[1], pts)
    text11 = ''
    if position < 0:
        text11 = "Vehicle is {:.3f} Metre left of center".format(-position)
    else:
        text11 = "Vehicle is {:.3f} Metre right of center".format(position)
    
    # Put text on an image
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(result, text1, (10,30), font, 1, (255,255,255), 2)
    cv2.putText(result, text11, (10,60), font, 1, (255,255,255), 2)
    cv2.putText(result, text2, (10,100), font, 1, (255,255,255), 2)
    
    return result

#-----------------------------------------------------------------------#
# Function to generate a warped thresholded color binary image from the given
# color image and other threshold parameters
#-----------------------------------------------------------------------#
def getBinaryImage(img, mtx, dist, sobel_kernel=3, s_thresh=(170, 255), sx_thresh=(20, 100)):
    img1 = np.copy(img)
#    img = np.copy(img)
    
    img = cv2.undistort(img, mtx, dist, None, mtx)
#    plt.title('Inprogress image in the pipeline - Undistorted Image')
#    plt.imshow(img)
#    plt.show()
    undist = np.copy(img)
    

    img_size = (img.shape[1], img.shape[0])
    src, dst = getTransformationPoints(img)
    #pts = np.array([[10,5],[20,30],[70,20],[50,10]], np.int32)
#    src = src.reshape((-1,1,2))
#    src1 = src.astype(np.int32)
#    cv2.polylines(img,[src1],True,(255,0,0), 2)
    
    M, Minv = getTransformationMatrices(src, dst)
    warped = cv2.warpPerspective(img, M, img_size, flags=cv2.INTER_LINEAR)
#    visualize2Images(img, warped)
    img = warped
    #gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # Convert to HLS color space and separate the V channel
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS).astype(np.float)
    h_channel = hls[:,:,0]
    l_channel = hls[:,:,1]
    s_channel = hls[:,:,2]
    #s_channel = gray
    
    # Sobel x
    gradx = abs_sobel_thresh(s_channel, orient='x', sobel_kernel=sobel_kernel, thresh=(20, 100))
    grady = abs_sobel_thresh(s_channel, orient='y', sobel_kernel=sobel_kernel, thresh=(20, 100))
    mag_binary = mag_thresh(s_channel, sobel_kernel=sobel_kernel, mag_thresh=(20, 100))
    dir_binary = dir_threshold(s_channel, sobel_kernel=sobel_kernel, dir_thresh=(0.7, 1.3))
    combined  = np.zeros_like(dir_binary)
    combined [((gradx == 1) & (grady == 1)) | ((mag_binary == 1) & (dir_binary == 1))] = 1
    
    # Threshold color channel
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= s_thresh[0]) & (s_channel <= s_thresh[1])] = 1

    # Stack each channel
    # Note color_binary[:, :, 0] is all 0s, effectively an all black image. It might
    # be beneficial to replace this channel with something else.
    #color_binary = np.dstack(( img[:,:,1], combined, s_binary))
    #color_binary = np.dstack(( np.zeros_like(combined), combined, s_binary))
    color_binary = np.zeros_like(combined)
    color_binary[(s_binary > 0) | (combined > 0)] = 1
    #visualize3Images(img1, warped, color_binary)
    return M, Minv, undist, color_binary

#-----------------------------------------------------------------------#
# Define a class to store & send initial settings
#-----------------------------------------------------------------------#
class Parameters():
    def __init__(self):
        self.sobel_kernel=7             # Kernal Size to use in Sobel Processing
        self.x_thresh = (20, 100)       # Threshold for Sobel X-Orientation processing
        self.y_thresh = (20, 100)       # Threshold for Sobel Y-Orientation processing
        self.mag_thresh = (20, 100)     # Threshold for Sobel Magnitude processing
        self.dir_thresh = (0.7, 1.3)    # Threshold for Sobel Directional processing
        self.s_thresh=(170, 255)        # Threshold for Sobel Combined X Processed image
        self.sx_thresh=(20, 100)        # Threshold for Color Binary Processing
        self.M = None                   # Perspective Transformation Matrix
        self.Minv = None                # Perspective Transformation Inverse Matrix

    def printValues(self):
        print('sobel_kernel=', self.sobel_kernel )
        print('x_thresh=', self.x_thresh)
        print('y_thresh=', self.y_thresh)
        print('mag_thresh=', self.mag_thresh)
        print('dir_thresh=', self.dir_thresh)
        print('s_thresh=', self.s_thresh)
        print('sx_thresh=', self.sx_thresh)
        print('M=', self.M)
        print('Minv=', self.Minv)
        
#-----------------------------------------------------------------------#
# Define a class to store Camera Calibration settings
#-----------------------------------------------------------------------#
class CameraCalibration():
    def __init__(self):
        self.pickle_file_name = 'cam_calib_mtx_dist.p'  # Camera Calibration Details Pickel file name
        self.nx = 9                     # Number of inside corners in x axis
        self.ny = 6                     # Number of inside corners in y axis
        self.mtx = None                 # Camera Calibration Matrix
        self.dist = None                # Camera Distortion Matrix

#-----------------------------------------------------------------------#
# The image/frame processing pipeline function.  It intakes a color 
# image as an input and returns a color image containing a filled polygon
# for the lanes identifed.
#-----------------------------------------------------------------------#
def pipeline(img, mtx, dist, sobel_kernel=3, s_thresh=(170, 255), sx_thresh=(20, 100)):
    img_size = (img.shape[1], img.shape[0])
    global cnt_h
    global cnt_nh

    M, Minv, undist, binary_warped = getBinaryImage(img, mtx, dist, sobel_kernel, s_thresh=s_thresh, sx_thresh=sx_thresh)
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
    leftx, lefty, rightx, righty = None, None, None, None

    if (left_line.detected == False) or (right_line.detected == False):
        #find by histogram method
        print('to detect lane through histogram')
        leftx, lefty, rightx, righty = findXY_Histogram(binary_warped)
        cnt_h += 1
    else:
        #find by non-histogram method
        print('to detect lane through non-histogram')
        leftx, lefty, rightx, righty = findXY_NonHistogram(binary_warped, left_line.current_fit, right_line.current_fit)
        cnt_nh += 1

    print('Lane finding by Histogram =', cnt_h, 'by Non-Histogram = ', cnt_nh)
    left_fit, right_fit, left_fitx, right_fitx, ploty = pixelPositionToXYValues(leftx, lefty, rightx, righty, binary_warped.shape[0])
    #Find Radius of Curvature
    left_curve_rad = calculateRadiusOfCurvature(leftx, lefty)
    right_curve_rad = calculateRadiusOfCurvature(rightx, righty)
    average_curve_rad = (left_curve_rad + right_curve_rad) / 2
    #print('Left', left_curve_rad, 'metres, Right', right_curve_rad, 'metres', 'Average-smoothed', average_curve_rad)
    
    print('left_fitx & right_fitx Before lineCheck', left_fitx[0], right_fitx[0])
    left_fit_x = lineCheck(left_line, left_curve_rad, left_fitx, left_fit)
    right_fit_x = lineCheck(right_line, right_curve_rad, right_fitx, right_fit)
    print('left_fitx & right_fitx After lineCheck', left_fitx[0], right_fitx[0])
    
    out_img[lefty, leftx] = [255, 0, 0]
    out_img[righty, rightx] = [0, 0, 255]
    #visualize2Images(binary_warped, out_img)
    
    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))
    roc_text = "Radius of Curvature: {0:.3f} Metres (Left={1:.3f}, Right={2:.3f})".format(average_curve_rad, left_curve_rad, right_curve_rad)
    cnt_text = ''#"By Histogram:{}, Non-Histogram:{}".format(cnt_h, cnt_nh)
    result = drawPolygonAndUnwrap(binary_warped, undist, pts, Minv, img_size, roc_text, cnt_text)
    return result


# Pipeline to be used for Image processing
#def pipeline(img, mtx, dist, sobel_kernel=3, s_thresh=(170, 255), sx_thresh=(20, 100)):
#    
#    img_size = (img.shape[1], img.shape[0])
#    M, Minv, undist, binary_warped = getBinaryImage(img, mtx, dist, sobel_kernel, s_thresh=s_thresh, sx_thresh=sx_thresh)
#    out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
#    leftx, lefty, rightx, righty, out_img = findXY_Histogram(binary_warped)
#    left_fit, right_fit, left_fitx, right_fitx, ploty = pixelPositionToXYValues(leftx, lefty, rightx, righty, binary_warped.shape[0])    
#    #Find Radius of Curvature
#    left_curve_rad = calculateRadiusOfCurvature(leftx, lefty)
#    right_curve_rad = calculateRadiusOfCurvature(rightx, righty)
#    average_curve_rad = (left_curve_rad + right_curve_rad) / 2
#    print('Left', left_curve_rad, 'metres, Right', right_curve_rad, 'metres', 'Average-smoothed', average_curve_rad)
#    
#    # Find camera position
#    left_mean = np.mean(leftx)
#    right_mean = np.mean(rightx)
#    camera_pos = (img_size[0]/2)-np.mean([left_mean, right_mean])
#    print('camera_pos', camera_pos )
#    xm_per_pix = 3.7/500
#    print('actual camera_pos', str(camera_pos*xm_per_pix)[:6], str(camera_pos*xm_per_pix))
#    
#    out_img[lefty, leftx] = [255, 0, 0]
#    out_img[righty, rightx] = [0, 0, 255]
#    
#    # Recast the x and y points into usable format for cv2.fillPoly()
#    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
#    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
#    pts = np.hstack((pts_left, pts_right))
#    roc_text = "Radius of Curvature: {0:.3f} Metres (Left={1:.3f}, Right={2:.3f})".format(average_curve_rad, left_curve_rad, right_curve_rad)
#    result = drawPolygonAndUnwrap(binary_warped, undist, pts, Minv, img_size, roc_text, None)
#    
#    return result
    
#-----------------------------------------------------------------------#
# A convenience method to process test images
#-----------------------------------------------------------------------#
def processImages():
    nx = 9  #Number of inside corners in x axis
    ny = 6  #Number of inside corners in y axis
    mtx, dist = calibrateCameraBySamples('camera_cal', 'calibration*.jpg', nx, ny, 'cam_calib_mtx_dist.p', True)
    #testUndistor('camera_cal/calibration5.jpg', mtx, dist)
    #testUndistor('test_images/test3.jpg', mtx, dist)
    ksize = 7 # Choose a Sobel Kernel size, larger odd number to smooth gradient measurements

    # Main Function
    print('\n***** Starting Main Function *****')
    cam_calib = CameraCalibration()
    cam_calib.mtx, cam_calib.dist = calibrateCameraBySamples('camera_cal', 'calibration*.jpg', cam_calib.nx, cam_calib.ny, cam_calib.pickle_file_name, True)
    #testUndistor('camera_cal/calibration5.jpg', cam_calib.mtx, cam_calib.dist)
    
    params = Parameters()
    params.camera_calibration = cam_calib
    params.printValues()
#    left_line = Line()
#    right_line = Line()
#    cnt_h = 0
#    cnt_nh = 0

    input_folder = 'test_images'
    output_folder = 'output_images'
    file_pattern = '*.jpg'
    fnames = os.path.join(input_folder, file_pattern)
    print('fnames', fnames)
    #fnames = 'camera_cal/calibration5.jpg'
    
    images = glob.glob(fnames)
    for idx, fname in enumerate(images):
        print(idx, fname)
        image = mpimg.imread(fname)
        result = pipeline(image, mtx, dist, sobel_kernel=ksize, s_thresh=(170, 255), sx_thresh=(20, 100))
        out_file = os.path.join(output_folder, os.path.split(fname)[1])
        mpimg.imsave(out_file, result)
        plt.figure(figsize=(16,12))
        plt.imshow(result)
        plt.show()
        print()
    return True

#-----------------------------------------------------------------------#
# A convenience method to process video frame
#-----------------------------------------------------------------------#
def processVideoFrame(image):
    result = pipeline(image, cam_calib.mtx, cam_calib.dist, sobel_kernel=params.sobel_kernel, s_thresh=params.s_thresh, sx_thresh=params.sx_thresh)
    return result

#-----------------------------------------------------------------------#
# A convenience method to process video file
#-----------------------------------------------------------------------#
def processVideo():
    # Main Function
    print('\n***** Starting Main Function *****')
#    cam_calib = CameraCalibration()
#    cam_calib.mtx, cam_calib.dist = calibrateCameraBySamples('camera_cal', 'calibration*.jpg', cam_calib.nx, cam_calib.ny, cam_calib.pickle_file_name, True)
#    #testUndistor('camera_cal/calibration5.jpg', cam_calib.mtx, cam_calib.dist)
#    
#    params = Parameters()
#    params.camera_calibration = cam_calib
#    params.printValues()
#    left_line = Line()
#    right_line = Line()
#    cnt_h = 0
#    cnt_nh = 0
    
    #vc_in_fn = 'NH_45_NearChennai.mp4'
    #vc_in_fn = 'harder_challenge_video.mp4'
    #vc_in_fn = 'challenge_video.mp4'
    vc_in_fn = 'project_video.mp4'
    vc_out_fn = 'out_' + vc_in_fn
    vclip = VideoFileClip(vc_in_fn)
    #vclip = vclip.subclip(0, 20)
    processed_vclip = vclip.fl_image(processVideoFrame)
    processed_vclip.write_videofile(vc_out_fn, audio=False)
    
    print('***** Program Execution Completed *****\n')
    return True

cam_calib = CameraCalibration()
cam_calib.mtx, cam_calib.dist = calibrateCameraBySamples('camera_cal', 'calibration*.jpg', cam_calib.nx, cam_calib.ny, cam_calib.pickle_file_name, True)
#testUndistor('camera_cal/calibration5.jpg', cam_calib.mtx, cam_calib.dist)

params = Parameters()
params.camera_calibration = cam_calib
params.printValues()
cnt_h = 0
cnt_nh = 0
left_line = Line()
right_line = Line()

processImages()
#processVideo()

#mtx, dist = calibrateCameraBySamples('camera_cal', 'calibration*.jpg', 9, 6, pickle_file_name='cam_calib_mtx_dist.p')
#testUndistor('camera_cal/calibration5.jpg', mtx, dist)
#testUndistor('test_images/test2.jpg', mtx, dist)