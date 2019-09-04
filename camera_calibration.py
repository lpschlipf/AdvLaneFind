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

    :return: Calibration Matrix and Distortion coefficients
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
            obj_points.append(objp)
            img_points.append(corners)

            # Draw and display the corners
            # cv2.drawChessboardCorners(img, (nx, ny), corners, ret)
            # plt.imshow(img)
        else:
            print("Calibration failed on image {}".format(image))

    # Calculate the Camera Calibration Matrix
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points, gray_shape, None, None)
    return mtx, dist


def undist_image(img, mtx, dist):
    """ Undistort an image based on a calibration matrix and distortion coefficients

    :param img: Distorted image
    :param mtx: Calibration matrix
    :param dist: Distortion coefficients
    :return: Undistorted image
    """
    return cv2.undistort(img, mtx, dist, None, mtx)


def warp_to_birdseye(img, calibrate=True):
    """ Unwarp and image to a top down birds eye perspective.

    :param img: Image to warp
    :param calibrate: Boolean parameter that turns on plotting of source and destination points
    :return: unwarped image
    """
    y, x = img.shape
    x_offset, y_offset = 150, 50
    # define 4 source points src = np.float32([[,],[,],[,],[,]])
    # these points were defined by optical inspection of an image with straight lines
    src = np.float32([[x/2 + 66, y - 260],
                      [x / 2 - 65, y - 260],
                      [x/2 - 410, y - 30],
                      [x/2 + 435, y - 30]])
    # define 4 destination points dst = np.float32([[,],[,],[,],[,]])
    dst = np.float32([[x - x_offset, y_offset],
                      [x_offset, y_offset],
                      [x_offset, y - y_offset],
                      [x - x_offset, y - y_offset]])
    # d) use cv2.getPerspectiveTransform() to get M, the transform matrix
    M = cv2.getPerspectiveTransform(src, dst)
    M_inv = cv2.getPerspectiveTransform(dst, src)

    # warp image
    warped = warp(img, M)

    if calibrate:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
        ax1.imshow(img, cmap='gray')
        ax1.plot(np.append(src[:, 0], src[0, 0]), np.append(src[:, 1], src[0, 1]),
                 linewidth=1.0, marker='x', ms=10, color='r')
        ax1.set_title("Unwarped image with source points")
        ax2.imshow(warped, cmap='gray')
        ax2.plot(np.append(dst[:, 0], dst[0, 0]), np.append(dst[:, 1], dst[0, 1]),
                 marker='x', ms=10.0, color='r')
        ax2.set_title("Warped image with destination points")
    return warped, M, M_inv


def color_threshold(img, s_thresh=(120, 255), sx_thresh=(20, 100)):
    """Perform a color and gradient thresholding and return a binary image for the chosen parameters.
    The image is transformed to HLS color space.
    An edge detection using the Sobel operator in x direction with a threshold is performed.
    Also a color threshold is applied in the s channel of the HLS color space.
    Finally these two binary images are combined with a logical or operation.

    :param img: RGB Image that shall be converted.
    :param s_thresh: Parameters for color threshold.
    :param sx_thresh: Parameters for Sobel threshold.
    :return: Stacked results from both pipelines and combined result from both pipelines.
    """
    img = np.copy(img)
    # Convert to HLS color space and separate the S channel.
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


def warp(img, M):
    """ Warp an image given a transformation matrix.

    :param img: image to waro
    :param M: transformation matrix
    :return: transformed image
    """
    img_size = (img.shape[1], img.shape[0])
    return cv2.warpPerspective(img, M, img_size, flags=cv2.INTER_LINEAR)


if __name__ == '__main__':
    from lane_finder import fit_poly
    nx, ny = 9, 6
    # Calibrate Camera on checkerboard images in the folder camera_cal
    mtx, dist = cal_cam()
    print("matrix: {} ; distortion: {}".format(mtx, dist))
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
    birdseye, M, Minv = warp_to_birdseye(binary)

    from lane_finder import LaneFinder
    lf = LaneFinder()
    # Feed lanefinder previously found fit values
    # lf.left_fit = np.array([0, 0, 1.60436725e+02])
    # lf.right_fit = np.array([0, 0, 1.13855949e+03])
    lane_fit_img = lf.find_lane(birdseye, visualization=True)
    birdseye_with_rad, left_curverad, right_curverad = lf.measure_lane_geometry(birdseye)
    print('Curvature in m: {}'.format((left_curverad, right_curverad)))

    # Plot Undistorted Image

    # f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
    # f.tight_layout()
    # ax1.imshow(img)
    # ax1.set_title('Original Image')
    # ax2.imshow(undist_img)
    # ax2.set_title('Undistorted Image')
    # plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)

    # Plot Threshold Image

    # f, ax3 = plt.subplots()
    # ax3.imshow(color_binary)
    # ax3.set_title("Image after color and gradient thresholding, color encoded")

    # Plot Birdseye Image

    # f, ax4 = plt.subplots()
    # ax4.imshow(birdseye, cmap="gray")
    # ax4.set_title("Binary image warped to bidseye view.")

    plt.show()
