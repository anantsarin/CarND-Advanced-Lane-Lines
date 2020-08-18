
#for image = 1
# ADVANCE LANE DETECTION
###################################################one time  ###################
#importing some useful packages
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import glob
import pickle
%matplotlib inline

# Read test images for camera caliberation

import os
os.listdir("camera_cal/")

# caliberation dimentions for chess board
chess_x = 9
chess_y = 6


# prepare object points

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((chess_y*chess_x,3), np.float32)
objp[:,:2] = np.mgrid[0:chess_x, 0:chess_y].T.reshape(-1,2)

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d points in real world space
imgpoints = [] # 2d points in image plane.

# Make a list of calibration images
images = glob.glob('camera_cal/calibration*.jpg')
# Step through the list and search for chessboard corners
for idx, fname in enumerate(images):
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Find the chessboard corners
    ret, corners = cv2.findChessboardCorners(gray, (chess_x,chess_y), None)
    # If found, add object points, image points
    if ret == True:
        objpoints.append(objp)
        imgpoints.append(corners)
        # Draw and display the corners
        cv2.drawChessboardCorners(img, (chess_x,chess_y), corners, ret)
        #write_name = 'corners_found'+str(idx)+'.jpg'
        #cv2.imwrite(write_name, img)
        #plt.imshow(img)


def cal_undistort(img, objpoints, imgpoints):
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img.shape[1:], None, None)
    undist = cv2.undistort(img, mtx, dist, None, mtx)
    return undist, mtx, dist


#### Test distortion

# Test undistortion on an calibration1.jpg
img1 = mpimg.imread('camera_cal/calibration1.jpg')
undist, mtx, dist = cal_undistort(img1, objpoints, imgpoints)
#dst = cv2.cvtColor(dst, cv2.COLOR_BGR2RGB)
# Visualize undistortion
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20,10))
ax1.imshow(img1)
ax1.set_title('Original Image', fontsize=30)
ax2.imshow(undist)
ax2.set_title('Undistorted Image', fontsize=30)


####################################################


def warper(img, src, dst):
    # Compute and apply perpective transform
    img_size = (img.shape[1], img.shape[0])
    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(img, M, img_size, flags=cv2.INTER_NEAREST)  # keep same size as input image
    return warped

def region_of_interest(img, vertices):
    """
    Applies an image mask.

    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    `vertices` should be a numpy array of integer points.
    """
    #defining a blank mask to start with
    mask = np.zeros_like(img)

    #defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255

    #filling pixels inside the polygon defined by "vertices" with the fill color
    cv2.fillPoly(mask, vertices, ignore_mask_color)

    #returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image


def find_lane_pixels(binary_warped, nwindows = 9, margin = 100, minpix = 50):
    # HYPERPARAMETERS
    # Choose the number of sliding windows
    # Set the width of the windows +/- margin
    # Set minimum number of pixels found to recenter window


    print (binary_warped.shape[0], binary_warped.shape[1])
    # Take a histogram of the bottom half of the image
    histogram = np.sum(binary_warped[binary_warped.shape[0]//2:,:], axis=0)
    # Create an output image to draw on and visualize the result
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))
    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0]//2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    # Set height of windows - based on nwindows above and image shape
    window_height = np.int(binary_warped.shape[0]//nwindows)
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # Current positions to be updated later for each window in nwindows
    leftx_current = leftx_base
    rightx_current = rightx_base

    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []
    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = binary_warped.shape[0] - (window+1)*window_height
        win_y_high = binary_warped.shape[0] - window*window_height


        ### TO-DO: Find the four below boundaries of the window ###
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin
        print (win_xleft_low, win_xleft_high, win_xright_low, win_xright_high)

        # Draw the windows on the visualization image
#         cv2.rectangle(out_img, (win_xleft_low,win_y_low), (win_xleft_high,win_y_high), (0,255,0), 2)
#         cv2.rectangle(out_img, (win_xright_low,win_y_low), (win_xright_high,win_y_high), (0,255,0), 2)

        ### TO-DO: Identify the nonzero pixels in x and y within the window ###
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) &  (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) &  (nonzerox < win_xright_high)).nonzero()[0]

        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)

        ### If you found > minpix pixels, recenter next window ###
        ### (`right` or `leftx_current`) on their mean position ###\
        if len(good_left_inds) > minpix:
            leftx_current = int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:
            rightx_current = int(np.mean(nonzerox[good_right_inds]))
        # Remove this when you add your function

    # Concatenate the arrays of indices (previously was a list of lists of pixels)
    try:
        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)
    except ValueError:
        # Avoids an error if the above is not implemented fully
        pass

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    return leftx, lefty, rightx, righty, out_img



def fit_poly(img_shape, leftx, lefty, rightx, righty):
    ### Fit a second order polynomial to each with np.polyfit() ###

    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)
    print (left_fit)
    print (right_fit)
    # Generate x and y values for plotting
    ploty = np.linspace(0, img_shape-1, img_shape)
    ### TO-DO: Calc both polynomials using ploty, left_fit and right_fit ###

    left_fitx = (left_fit[0]*ploty**2) + (left_fit[1]*ploty) + left_fit[2]
    right_fitx = (right_fit[0]*ploty**2) + (right_fit[1]*ploty) + right_fit[2]

    return left_fit, right_fit, left_fitx, right_fitx, ploty

def weighted_img(img, initial_img, α=0.8, β=1., γ=0.):
    """
    initial_img * α + img * β + γ
    NOTE: initial_img and img must be the same shape!
    """
    return cv2.addWeighted(initial_img, α, img, β, γ)


def pipeline(img, s_thresh=(170, 255), sx_thresh=(20, 100)):
    img = np.copy(img)
    # Convert to HLS color space and separate the V channel
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    l_channel = hls[:,:,1] # or h_channel = hls[:,:,1]
    s_channel = hls[:,:,2]
    # Sobel x
    sobelx = cv2.Sobel(l_channel, cv2.CV_64F, 1, 0) # Take the derivative in x
    abs_sobelx = np.absolute(sobelx) # Absolute x derivative to accentuate lines away from horizontal
    scaled_sobel = np.uint8(255*abs_sobelx/np.max(abs_sobelx))

    # Threshold x gradient
    sxbinary = np.zeros_like(scaled_sobel)
    sxbinary[(scaled_sobel >= sx_thresh[0]) & (scaled_sobel <= sx_thresh[1])] = 1

    # Threshold color channel
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= s_thresh[0]) & (s_channel <= s_thresh[1])] = 1
    # Stack each channel
    color_binary = np.dstack(( np.zeros_like(sxbinary), sxbinary, s_binary)) * 255
    combined_binary = np.zeros_like(sxbinary)
    combined_binary[(s_binary == 1) | (sxbinary == 1)] = 1
    return combined_binary



def work_with_bird_eye_view_image(img_pipeline):
    # Find our lane pixels first
    leftx, lefty, rightx, righty, out_img = find_lane_pixels(img_pipeline)
    # get line coefficients and line pixels
    left_fit, right_fit, left_fitx, right_fitx, ploty = fit_poly(img_pipeline.shape[0], leftx, lefty, rightx, righty)

    # Colors in the left and right lane regions
    out_img[lefty, leftx] = [255, 0, 0]
    out_img[righty, rightx] = [0, 0, 255]
    # Plots the left and right polynomials on the lane lines
    # plt.plot(left_fitx, ploty, color='yellow')
    # plt.plot(right_fitx, ploty, color='yellow')

    print ( "curvature  = " , left_fitx[719]- right_fitx[719])
    ## Visualization ##
    # Create an image to draw on and an image to show the selection window
    blank_img = np.dstack((img_pipeline, img_pipeline, img_pipeline))*255
    window_img = np.zeros_like(blank_img)
    # Color in left and right line pixels
    blank_img[lefty, leftx] = [255, 0, 0]
    blank_img[righty, rightx] = [0, 0, 255]
    # blank_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
    # blank_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]
    # Generate a polygon to illustrate the search window area
    # And recast the x and y points into usable format for cv2.fillPoly()
    left_line = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    right_line = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    line_pts = np.hstack((right_line, left_line ))


    # Draw the lane onto the warped blank image
    cv2.fillPoly(window_img, np.int_([line_pts]), (0,255, 0))
    result = cv2.addWeighted(blank_img, 1, window_img, 0.3, 0)


    return left_fit, right_fit, ploty, result

def measure_curvature_real_center(lane_values, left_fit, right_fit, ploty):
    '''
    Calculates the curvature of polynomial functions in meters.
    '''
    # Define conversions in x and y from pixels space to meters
    ym_per_pix = 30/720 # meters per pixel in y dimension
    xm_per_pix = 3.7/lane_values #700 # meters per pixel in x dimension

    # Define y-value where we want radius of curvature
    # We'll choose the maximum y-value, corresponding to the bottom of the image
    y_eval = np.max(ploty)*ym_per_pix

    ##### TO-DO: Implement the calculation of R_curve (radius of curvature) #####
    left_curverad = ((1+(2*left_fit[0]*y_eval + left_fit[1])**2)**1.5)/np.absolute(2*left_fit[0])
    right_curverad = ((1+(2*right_fit[0]*y_eval + right_fit[1])**2)**1.5)/np.absolute(2*right_fit[0])
    print (left_curverad, right_curverad)
    avg_curv = (left_curverad+ right_curverad)/2
    center_of_image = 640 # Let camera is at the center of the car
    distance_from_center = abs((center_of_image - lane_values)*xm_per_pix)
    return distance_from_center, left_curverad, right_curverad


# # Run image through the pipeline
# # Note that in your project, you'll also want to feed in the previous fits
# lane_poly = search_around_poly(result)

# # View your output
# plt.imshow(lane_poly)


def search_around_poly(binary_warped, left_fit, right_fit):
    # HYPERPARAMETER
    # Choose the width of the margin around the previous polynomial to search
    # The quiz grader expects 100 here, but feel free to tune on your own!
    margin = 100

    # Grab activated pixels
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])

    ### Set the area of search based on activated x-values ###
    ### within the +/- margin of our polynomial function ###
    left_lane_neg_margin = ((left_fit[0]*nonzeroy**2) + (left_fit[1]*nonzeroy) + left_fit[2] - margin)
    left_lane_pos_margin = ((left_fit[0]*nonzeroy**2) + (left_fit[1]*nonzeroy) + left_fit[2] + margin)
    right_lane_neg_margin = ((right_fit[0]*nonzeroy**2) + (right_fit[1]*nonzeroy) + right_fit[2] - margin)
    right_lane_pos_margin = ((right_fit[0]*nonzeroy**2) + (right_fit[1]*nonzeroy) + right_fit[2] + margin)
    left_lane_inds = ((nonzerox >= left_lane_neg_margin) & (nonzerox <= left_lane_pos_margin))
    right_lane_inds = ((nonzerox >= right_lane_neg_margin) & (nonzerox <= right_lane_pos_margin))


    left_lane_inds = ((nonzerox > (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + left_fit[2] - margin)) & (nonzerox < (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + left_fit[2] + margin)))
    right_lane_inds = ((nonzerox > (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy +  right_fit[2] - margin)) & (nonzerox < (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + right_fit[2] + margin)))

    # Again, extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    # Fit new polynomials
    # left_fit, right_fit, left_fitx, right_fitx, ploty = fit_poly(binary_warped.shape, leftx, lefty, rightx, righty)

    return leftx, lefty, rightx, righty



def main_one_image():

    # Read Image

#     image = mpimg.imread('test_images/test2.jpg')
    image = mpimg.imread('test_images/straight_lines1.jpg')

    #printing out some stats and plotting
    print('This image is:', type(image), 'with dimensions:', image.shape)
    plt.imshow(image)
    # if you wanted to show a single color channel image called 'gray', for example, call as plt.imshow(gray, cmap='gray')

    undist, _, _ = cal_undistort(img1, objpoints, imgpoints)
    # for region of interest
    vertices = np.array([[(190,720),(600, 410),(670, 410), (1100,720)]],dtype=np.int32)
    img_roi = region_of_interest(undist, vertices)
    ################################
    img_pipeline = pipeline(img_roi, s_thresh=(140, 200), sx_thresh=(10, 140))
    plt.imshow(img_pipeline)
    plt.show()
    # warp to bird eye view
    src_for_warp = np.array([[(196,720),(600, 450),(680, 450), (1135,720)]],dtype=np.float32)
    dst_for_warp = np.array([[(320,720),(320, 0),(960, 0),(960,720)]], dtype=np.float32)
    warped = warper(img_pipeline, src_for_warp, dst_for_warp)
    # Apply thresh hold and gradient



    # warp to bird eye view
#     src_for_warp = np.array([[(196,720),(600, 450),(680, 450), (1135,720)]],dtype=np.float32)
#     dst_for_warp = np.array([[(320,720),(320, 0),(960, 0),(960,720)]], dtype=np.float32)
#     warped = warper(image, src_for_warp, dst_for_warp)
#     # Apply thresh hold and gradient
#     img_pipeline = pipeline(warped, s_thresh=(140, 200), sx_thresh=(10, 140))
    # fit lane line on image
    ###########################################################################################
    # left_fit, right_fit, ploty, out_img = work_with_bird_eye_view_image(img_pipeline)
    # Find our lane pixels first
    leftx, lefty, rightx, righty, out_img = find_lane_pixels(warped)
    # get line coefficients and line pixels
    left_fit, right_fit, left_fitx, right_fitx, ploty = fit_poly(warped.shape[0], leftx, lefty, rightx, righty)

    # Colors in the left and right lane regions
#     out_img[lefty, leftx] = [255, 0, 0]
#     out_img[righty, rightx] = [0, 0, 255]
    # Plots the left and right polynomials on the lane lines
    # plt.plot(left_fitx, ploty, color='yellow')
    # plt.plot(right_fitx, ploty, color='yellow')

    print ( "curvature  = " , left_fitx[719]- right_fitx[719])
    ## Visualization ##
    # Create an image to draw on and an image to show the selection window
    out_img_min = np.dstack((warped, warped, warped))*255
    out_img = np.zeros_like(out_img_min)
    window_img = np.zeros_like(out_img)
    # Color in left and right line pixels
    out_img[lefty, leftx] = [255, 0, 0]
    out_img[righty, rightx] = [0, 0, 255]
    plt.imshow(out_img)
    plt.show()
    # blank_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
    # blank_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]
    # Generate a polygon to illustrate the search window area
    # And recast the x and y points into usable format for cv2.fillPoly()
    left_line = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    right_line = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    line_pts = np.hstack((right_line, left_line ))


    # Draw the lane onto the warped blank image
    cv2.fillPoly(window_img, np.int_([line_pts]), (0,255, 0))
    plt.imshow(window_img)
    plt.show()
    result = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)


    # return left_fit, right_fit, ploty, result
    plt.imshow(result)
    plt.show()

    # calculate curvature for the lines
    # Calculate the radius of curvature in meters for both lane lines
    print ( "curvature  = " ,  )
    distance_from_center, avg_curv = measure_curvature_real_center(int(right_fitx[719]-left_fitx[719]), left_fit, right_fit, ploty)

    print(avg_curv, 'm')

    # un-warp the image
    dst_unwarp = np.array([[(196,720),(600, 450),(680, 450), (1135,720)]],dtype=np.float32)
    src_unwarp = np.array([[(320,720),(320, 0),(960, 0),(960,720)]], dtype=np.float32)
    unwarped = warper(result, src_unwarp, dst_unwarp)

    #cv2.putText(unwarped, "AAAAAAA", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0,0,0))
    #cv2.putText(img_resized, text, (x,y), font, font_size, font_color, font_thickness, cv2.LINE_AA)

    plt.imshow(unwarped)
    plt.show()
    # add the calculated lines on the color image
    img2 = weighted_img(image, unwarped)
    cv2.putText(img2, "abcdddddfsdf", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2.1, (255,255,255), 4)
    plt.imshow(img2)
    plt.show()

    # Plot the result
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
    f.tight_layout()

    ax1.imshow(image)
    ax1.set_title('Original Image', fontsize=40)

    ax2.imshow(img2)
    ax2.set_title('Pipeline Result', fontsize=40)
    plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
    plt.show()

main_one_image()


class Line():
    def __init__(self):
        # was the line detected in the last iteration?
        self.detected = False
        #polynomial coefficients for the most recent fit
        self.current_fit = [np.array([False])]
        #radius of curvature of the line in some units
        self.radius_of_curvature_right = None
        self.radius_of_curvature_left = None
        #distance in meters of vehicle center from the line
        self.line_base_pos = None
        #number of iteration
        self.n = 0
        #reset after n iteration
        self.src_for_warp = np.array([[(196,720),(600, 450),(680, 450), (1135,720)]],dtype=np.float32)
        self.dst_for_warp = np.array([[(320,720),(320, 0),(960, 0),(960,720)]], dtype=np.float32)
        # avg left fit and avg right fit
        self.left_fit_avg = np.array([0,0,0])
        self.right_fit_avg = np.array([0,0,0])
        # x fits and y fits
        self.left_fitx = None
        self.right_fitx = None
        #


    def avg_line(self):
        self.left_fit_avg = (self.left_fit_avg*n + self.left_fitx)/(n+1)
        self.right_fit_avg = (self.right_fit_avg*n + self.right_fitx)/(n+1)
        # reset the average after every x frames
        if self.n == 75:
            # reset n
            self.n = 0



    def use_avg(self, distance_between_pix):
        #currently out distance between two  lines are arount 600 px
        #so we will remove the lines giving us wrong data
        if distance_between_pix >750 or distance_between_pix <= 450:
            return False
        return True



line_save = Line()

def one_image_processing(image):

    undist, _, _ = cal_undistort(img1, objpoints, imgpoints)
    # Apply thresh hold and gradient
    img_pipeline = pipeline(image, s_thresh=(140, 200), sx_thresh=(10, 140))
    # warp to bird eye view
    warped = warper(img_pipeline, line_save.src_for_warp, line_save.dst_for_warp)
    if line_save.detected:
        leftx, lefty, rightx, righty = search_around_poly(warped, left_fit, right_fit)
    else:
        # Find our lane pixels first
        leftx, lefty, rightx, righty, out_img = find_lane_pixels(warped)
        line_save.detected = True
        # get line coefficients and line pixels
    left_fit, right_fit, left_fitx, right_fitx, ploty = fit_poly(warped.shape[0], leftx, lefty, rightx, righty)
    px_distace_at_base = int(right_fitx[719]-left_fitx[719])
    # fill class
    is_data_correct = line_save.use_avg(px_distace_at_base)
    if line_save.use_avg(px_distace_at_base):
        left_fitx = line_save.left_fitx
        right_fitx = line_save.right_fitx
        left_fit = line_save.left_fit_avg
        right_fit = line_save.right_fit_avg

    line_save.left_fitx = left_fitx
    line_save.right_fitx = right_fitx
    line_save.left_fit = left_fit
    line_save.right_fit = right_fit
    line_save.calculate_line_avg()
    line_save.n += 1

    ## Visualization ##
    # Create an image to draw on and an image to show the selection window
    out_img_min = np.dstack((warped, warped, warped))*255
    out_img = np.zeros_like(out_img_min)
    window_img = np.zeros_like(out_img)

    # Color in left and right line pixels
    out_img[lefty, leftx] = [255, 0, 0]
    out_img[righty, rightx] = [0, 0, 255]

    # Generate a polygon to illustrate the search window area
    # And recast the x and y points into usable format for cv2.fillPoly()
    left_line = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    right_line = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    line_pts = np.hstack((right_line, left_line ))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(window_img, np.int_([line_pts]), (0,255, 0))
    result = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)

    # calculate curvature for the lines
    # Calculate the radius of curvature in meters for both lane lines
    distance_from_center, left_curv, right_curv = measure_curvature_real_center(px_distace_at_base, left_fit, right_fit, ploty)
    # un-warp the image
    unwarped = warper(result, line_save.dst_for_warp, line_save.src_for_warp)

    # add the calculated lines on the color image
    img2 = weighted_img(image, unwarped)
    side = ""
    if distance_from_center > 0:
        side = "right"
    else:
        side = "left"
    distance_text = "distance from center "+ str("{:.3f}".format(distance_from_center))+ " " + side +" (m)"

    curvature_text = "Radius of curvature " + str(int(left_curv)) + " L(m) " + str(int(right_curv))+ " R(m) "

    cv2.putText(img2, distance_text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255,255,255), 4)
    cv2.putText(img2, curvature_text, (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255,255,255), 4)

    return img2


















#################################################################




class Line():
    def __init__(self):
        # was the line detected in the last iteration?
        self.detected = False
        # x values of the last n fits of the line
        self.recent_xfitted = []
        #average x values of the fitted line over the last n iterations
        self.bestx = None
        #polynomial coefficients averaged over the last n iterations
        self.best_fit = None
        #polynomial coefficients for the most recent fit
        self.current_fit = [np.array([False])]
        #radius of curvature of the line in some units
        self.radius_of_curvature_right = None
        self.radius_of_curvature_left = None
        #distance in meters of vehicle center from the line
        self.line_base_pos = None
        #number of iteration
        self.n = 0
        #reset after n iteration
        self.src_for_warp = np.array([[(196,720),(600, 450),(680, 450), (1135,720)]],dtype=np.float32)
        self.dst_for_warp = np.array([[(320,720),(320, 0),(960, 0),(960,720)]], dtype=np.float32)
        # avg left fit and avg right fit
        self.left_fit_avg = np.array([0,0,0])
        self.right_fit_avg = np.array([0,0,0])
        # x fits and y fits
        self.left_fitx = np.array([0,0,0])
        self.right_fitx = np.array([0,0,0])
        #


    def calculate_line_avg(self):
        self.left_fit_avg = (self.left_fit_avg*self.n + self.left_fitx)/(self.n+1)
        self.right_fit_avg = (self.right_fit_avg*self.n + self.right_fitx)/(self.n+1)
        # reset the average after every x frames
        if self.n == 75:
            # reset n
            self.n = 0



    def use_avg(self, distance_between_pix):
        #currently out distance between two  lines are arount 600 px
        #so we will remove the lines giving us wrong data
        if distance_between_pix >750 or distance_between_pix <= 450:
            return False
        return True

line_save = Line()

def one_image_processing(image):

    # if you wanted to show a single color channel image called 'gray', for example, call as plt.imshow(gray, cmap='gray')
    undist, _, _ = cal_undistort(img1, objpoints, imgpoints)
    # for region of interest
    vertices = np.array([[(190,720),(600, 410),(670, 410), (1100,720)]],dtype=np.int32)
    img_roi = region_of_interest(undist, vertices)
    ################################
    img_pipeline = pipeline(image, s_thresh=(140, 200), sx_thresh=(10, 140))
    # warp to bird eye view
    warped = warper(img_pipeline, line_save.src_for_warp, line_save.dst_for_warp)
    # Apply thresh hold and gradient

    # fit lane line on image
    if line_save.detected:
        print ("in if")
        leftx, lefty, rightx, righty = search_around_poly(warped, left_fit, right_fit)
    else:
        print ("in else")
        # Find our lane pixels first
        leftx, lefty, rightx, righty, out_img = find_lane_pixels(warped)
        line_save.detected = True
        # get line coefficients and line pixels

    left_fit, right_fit, left_fitx, right_fitx, ploty = fit_poly(warped.shape[0], leftx, lefty, rightx, righty)
    print (left_fit, right_fit)
    px_distace_at_base = int(right_fitx[719]-left_fitx[719])
    # fill class
    is_data_correct = line_save.use_avg(px_distace_at_base)
    if is_data_correct:
        left_fitx = line_save.left_fitx
        right_fitx = line_save.right_fitx
        left_fit = line_save.left_fit_avg
        right_fit = line_save.right_fit_avg

    line_save.left_fitx = left_fitx
    line_save.right_fitx = right_fitx
    line_save.left_fit = left_fit
    line_save.right_fit = right_fit
    line_save.calculate_line_avg()
    line_save.n += 1

    # Colors in the left and right lane regions
    ## Visualization ##
    # Create an image to draw on and an image to show the selection window
    out_img_min = np.dstack((warped, warped, warped))*255
    out_img = np.zeros_like(out_img_min)
    window_img = np.zeros_like(out_img)

    # Color in left and right line pixels
    out_img[lefty, leftx] = [255, 0, 0]
    out_img[righty, rightx] = [0, 0, 255]

    # Generate a polygon to illustrate the search window area
    # And recast the x and y points into usable format for cv2.fillPoly()
    left_line = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    right_line = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    line_pts = np.hstack((right_line, left_line ))


    # Draw the lane onto the warped blank image
    cv2.fillPoly(window_img, np.int_([line_pts]), (0,255, 0))

    result = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)

    distance_from_center, left_curv, right_curv  = measure_curvature_real_center(px_distace_at_base, left_fit, right_fit, ploty)

    # un-warp the image
    unwarped = warper(result, line_save.dst_for_warp, line_save.src_for_warp)

    img2 = weighted_img(image, unwarped)
    side = ""
    if distance_from_center > 0:
        side = "right"
    else:
        side = "left"
    distance_text = "distance from center "+ str("{:.2f}".format(distance_from_center))+ " " + side +" (m)"

    curvature_text = "Radius of curvature " + str(int(left_curv)) + " L(m) " + str(int(right_curv))+ " R(m) "

    cv2.putText(img2, distance_text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255,255,255), 4)
    cv2.putText(img2, curvature_text, (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255,255,255), 4)


    return img2


















class Line():
    def __init__(self):
        # was the line detected in the last iteration?
        self.detected = False
        #radius of curvature of the line in some units
        self.radius_of_curvature_right = None
        self.radius_of_curvature_left = None
        #distance in meters of vehicle center from the line
        self.line_base_pos = None
        #number of iteration
        self.n = 0
        #reset after n iteration
        self.src_for_warp = np.array([[(196,720),(600, 450),(680, 450), (1135,720)]],dtype=np.float32)
        self.dst_for_warp = np.array([[(320,720),(320, 0),(960, 0),(960,720)]], dtype=np.float32)
        # avg left fit and avg right fit
        self.left_fit_avg = np.array([0,0,0])
        self.right_fit_avg = np.array([0,0,0])
        # x fits and y fits
        self.left_fitx = np.array([0,0,0])
        self.right_fitx = np.array([0,0,0])

    def calculate_line_avg(self):
        self.left_fit_avg = ((self.left_fit_avg*self.n) + self.left_fit)/(self.n+1)
        self.right_fit_avg = ((self.right_fit_avg*self.n) + self.right_fit)/(self.n+1)
        # reset the average after every x frames
        if self.n == 10:
            self.n = 0

    def is_data_correct(self, distance_between_pix):
        #currently out distance between two  lines are arount 600 px
        #so we will remove the lines giving us wrong data
        if distance_between_pix >750 or distance_between_pix <= 450:
            return False
        return True

line_save = Line()

def one_image_processing(image):

    undist, _, _ = cal_undistort(img1, objpoints, imgpoints)
    # for region of interest
    vertices = np.array([[(190,720),(600, 410),(670, 410), (1100,720)]],dtype=np.int32)
    img_roi = region_of_interest(undist, vertices)
    img_pipeline = pipeline(image, s_thresh=(140, 200), sx_thresh=(10, 140))
    # warp to bird eye view
    warped = warper(img_pipeline, line_save.src_for_warp, line_save.dst_for_warp)
    # Apply thresh hold and gradient

    # fit lane line on image
    if line_save.detected:
        leftx, lefty, rightx, righty = search_around_poly(warped, line_save.left_fit, line_save.right_fit)
    else:
        # Find our lane pixels first
        leftx, lefty, rightx, righty, out_img = find_lane_pixels(warped)
        line_save.detected = True
        # get line coefficients and line pixels

    left_fit, right_fit, left_fitx, right_fitx, ploty = fit_poly(warped.shape[0], leftx, lefty, rightx, righty)

    px_distace_at_base = int(right_fitx[719]-left_fitx[719])

    # fill Line class with variables
    is_data_correct = line_save.is_data_correct(px_distace_at_base)
    if not is_data_correct:
        left_fitx = line_save.left_fitx
        right_fitx = line_save.right_fitx
        left_fit = line_save.left_fit_avg
        right_fit = line_save.right_fit_avg
    # save data in line
    line_save.left_fitx = left_fitx
    line_save.right_fitx = right_fitx
    line_save.left_fit = left_fit
    line_save.right_fit = right_fit
    line_save.calculate_line_avg()
    line_save.n += 1

    # Colors in the left and right lane regions
    ## Visualization ##
    # Create an image to draw on and an image to show the selection window
    out_img_min = np.dstack((warped, warped, warped))*255
    out_img = np.zeros_like(out_img_min)
    window_img = np.zeros_like(out_img)

    # Color in left and right line pixels
    out_img[lefty, leftx] = [255, 0, 0]
    out_img[righty, rightx] = [0, 0, 255]

    # Generate a polygon to illustrate the search window area
    # And recast the x and y points into usable format for cv2.fillPoly()
    left_line = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    right_line = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    line_pts = np.hstack((right_line, left_line ))


    # Draw the lane onto the warped blank image
    cv2.fillPoly(window_img, np.int_([line_pts]), (0,255, 0))

    result = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)

    distance_from_center, left_curv, right_curv  = measure_curvature_real_center(px_distace_at_base, left_fit, right_fit, ploty)
    # un-warp the image
    unwarped = warper(result, line_save.dst_for_warp, line_save.src_for_warp)

    img2 = weighted_img(image, unwarped)
    side = ""
    if distance_from_center > 0:
        side = "right"
    else:
        side = "left"
    distance_text = "distance from center "+ str("{:.2f}".format(distance_from_center))+ " " + side +" (m)"
    curvature_text = "Radius of curvature " + str(int(left_curv)) + " L(m) " + str(int(right_curv))+ " R(m) "

    cv2.putText(img2, distance_text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255,255,255), 4)
    cv2.putText(img2, curvature_text, (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255,255,255), 4)
    return img2


######################################### Test function #########################
# Read Image

image = mpimg.imread('test_images/straight_lines1.jpg')
image_result = one_image_processing(image)
# Plot the result
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
f.tight_layout()

ax1.imshow(image)
ax1.set_title('Original Image', fontsize=40)

ax2.imshow(image_result)
ax2.set_title('Pipeline Result', fontsize=40)
plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
plt.show()