"""
Module to calibrate a camera from chessboard images and apply the respective transforms
"""
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob


def cal_cam(folder='camera_cal', nx=9, ny=6):
    """ Find camera calibration matrix from a set of chassboard images

    :return:
    """
    # Make a list of calibration images
    images = glob.glob(folder + '\calibration*.jpg')

    # Create the pointlists of measured and transformed image points to calculate the matrix
    img_points = []
    obj_points = []

    objp = np.zeros((nx*ny, 3), np.float32)
    objp[:, :2] = np.mgrid[0:nx, 0:ny].T.reshape(-1, 2)

    for image in images:
        img = cv2.imread(image)

        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray_shape = gray.shape[::-1]

        # Find the chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None, cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FILTER_QUADS)

        # If found, add points to arrays
        if ret == True:
            print('Calibration successful on {}'.format(image))
            obj_points.append(objp)
            img_points.append(corners)

            # Draw and display the corners
            # cv2.drawChessboardCorners(img, (nx, ny), corners, ret)
            # plt.imshow(img)

    # Calculate the Camera Calibration Matrix
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points, gray_shape, None, None)
    return mtx, dist


def undist_image(img, mtx, dist):
    """ Undistort an image based on a previously calculated distortion matrix

    :param img:
    :param mtx:
    :param dist:
    :return:
    """
    return cv2.undistort(img, mtx, dist, None, mtx)


def warp_perspective(img, nx=9, ny=5):
    """ Warp perspective of img to birdseye view.

    :param img:
    :param nx:
    :param ny:
    :return: Image with warped perspective.
    """
    # We need to detect 4 corners that define a rectangle here.
    src = np.float32([corners[0], corners[nx - 1], corners[-nx], corners[-1]])
    # c) define 4 destination points dst = np.float32([[,],[,],[,],[,]])
    y, x = img.shape
    offset = 100
    dst = np.float32([[offset, offset],
                      [x - offset, offset],
                      [offset, y - offset],
                      [x - offset, y - offset]])
    # d) use cv2.getPerspectiveTransform() to get M, the transform matrix
    M = cv2.getPerspectiveTransform(src, dst)
    # e) use cv2.warpPerspective() to warp your image to a top-down view
    img_size = (img.shape[1], img.shape[0])
    warped = cv2.warpPerspective(img, M, img_size, flags=cv2.INTER_LINEAR)
    return warped


def unwarp(img, calibrate=True):
    y, x = img.shape
    offset = 100
    # define 4 source points src = np.float32([[,],[,],[,],[,]])
    # these points were defined by optical inspection of an image with straight lines
    src = np.float32([[x/2 - 410, y - 30],
                      [x/2 + 435, y - 30],
                      [x/2 + 66, y - 260],
                      [x/2 - 65, y - 260]])
    # define 4 destination points dst = np.float32([[,],[,],[,],[,]])
    dst = np.float32([[x - offset, offset],
                      [offset, offset],
                      [offset, y - offset],
                      [x - offset, y - offset]])
    # d) use cv2.getPerspectiveTransform() to get M, the transform matrix
    M = cv2.getPerspectiveTransform(src, dst)
    # e) use cv2.warpPerspective() to warp your image to a top-down view
    img_size = (img.shape[1], img.shape[0])
    warped = cv2.warpPerspective(img, M, img_size, flags=cv2.INTER_LINEAR)
    if calibrate:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
        ax1.imshow(img)
        ax1.plot(src[:, 0], src[:, 1], linewidth=1.0, marker='x', ms=10, color='r')
        ax2.imshow(warped)
        ax2.plot(dst[:, 0], dst[:, 1], color='r')
    return warped, M


def color_threshold(img, s_thresh=(120, 255), sx_thresh=(20, 100)):
    img = np.copy(img)
    # Convert to HLS color space and separate the V channel
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    l_channel = hls[:, :, 1]
    s_channel = hls[:, :, 2]
    # Sobel x
    sobelx = cv2.Sobel(l_channel, cv2.CV_64F, 1, 0)  # Take the derivative in x
    abs_sobelx = np.absolute(sobelx)  # Absolute x derivative to accentuate lines away from horizontal
    scaled_sobel = np.uint8(255 * abs_sobelx / np.max(abs_sobelx))

    # Threshold x gradient
    sxbinary = np.zeros_like(scaled_sobel)
    sxbinary[(scaled_sobel >= sx_thresh[0]) & (scaled_sobel <= sx_thresh[1])] = 1

    # Threshold color channel
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= s_thresh[0]) & (s_channel <= s_thresh[1])] = 1
    # Stack each channel
    color_binary = np.dstack((np.zeros_like(sxbinary), sxbinary, s_binary)) * 255

    # Combine the two binary thresholds
    combined_binary = np.zeros_like(sxbinary)
    combined_binary[(s_binary == 1) | (sxbinary == 1)] = 1

    return combined_binary, color_binary


if __name__ == '__main__':
    nx, ny = 9, 6
    # Calibrate Camera on checkerboard images in the folder camera_cal
    mtx, dist = cal_cam()
    # Read test image and convert to RGB as pipeline expects!
    img = cv2.imread(r'C:\Code\AdvLaneFind\test_images\straight_lines1.jpg')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # Use calculated Matrix to undistort.
    undist_img = undist_image(img, mtx, dist)
    # Use the Sobel operator to calculate gradients and use a color threshold and combine
    # NOTE: for speed, get rid of colored binary output that shows which part of binary image is produced by what.
    binary, color_binary = color_threshold(undist_img)
    # Transform to Bird's eye perspective.
    # top_down, perspective_M = corners_unwarp(img, nx, ny, mtx, dist)
    birdseye, M = unwarp(binary)

    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
    f.tight_layout()
    ax1.imshow(img)
    ax1.set_title('Original Image', fontsize=50)
    ax2.imshow(undist_img)
    ax2.set_title('Undistorted Image', fontsize=50)
    plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)

    f, ax3 = plt.subplots()
    ax3.imshow(binary)

    f, ax4 = plt.subplots()
    ax4.imshow(birdseye)
    plt.show()
