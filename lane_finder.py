"""
This module contains methods to detect lanes from a binary birds eye image and to calculate their curvature.
"""

import numpy as np
import cv2
from matplotlib import pyplot as plt


def fit_poly(img_shape, leftx, lefty, rightx, righty):
    # Fit a second order Polynomial for left and right lane
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)
    # Generate x and y values for plotting
    ploty = np.linspace(0, img_shape[0] - 1, img_shape[0])
    # calculate values of polynomial on ploty
    left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
    right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]

    return left_fit, right_fit, left_fitx, right_fitx, ploty


def sliding_window_search(binary_warped, visualization=False):
    # Take a histogram of the bottom half of the image
    histogram = np.sum(binary_warped[binary_warped.shape[0] // 2:, :], axis=0)
    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0] // 2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    # HYPERPARAMETERS
    # Choose the number of sliding windows
    nwindows = 9
    # Set the width of the windows +/- margin
    margin = 100
    # Set minimum number of pixels found to recenter window
    minpix = 50

    # Set height of windows - based on nwindows above and image shape
    window_height = np.int(binary_warped.shape[0] // nwindows)
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

    # Create an output image to draw on and visualize the result
    out_img = np.dstack((binary_warped, binary_warped, binary_warped)) * 255

    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = binary_warped.shape[0] - (window + 1) * window_height
        win_y_high = binary_warped.shape[0] - window * window_height
        # find the four x boundaries of the window
        win_xleft_low = leftx_current - int(margin)
        win_xleft_high = leftx_current + int(margin)
        win_xright_low = rightx_current - int(margin)
        win_xright_high = rightx_current + int(margin)

        if visualization:
            # Draw the windows on the visualization image
            cv2.rectangle(out_img, (win_xleft_low, win_y_low),
                          (win_xleft_high, win_y_high), (0, 255, 0), 2)
            cv2.rectangle(out_img, (win_xright_low, win_y_low),
                          (win_xright_high, win_y_high), (0, 255, 0), 2)

        # identify the nonzero pixels in the image.
        good_left_inds = ((win_xleft_low <= nonzerox) & (nonzerox < win_xleft_high) \
                          & (win_y_low <= nonzeroy) & (nonzeroy < win_y_high)).nonzero()[0]
        good_right_inds = ((win_xright_low <= nonzerox) & (nonzerox < win_xright_high) \
                           & (win_y_low <= nonzeroy) & (nonzeroy < win_y_high)).nonzero()[0]

        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)

        # recenter the window if we found more then minpix
        if len(good_left_inds) > minpix:
            leftx_current = int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:
            rightx_current = int(np.mean(nonzerox[good_right_inds]))

    # Concatenate the arrays of indices (previously was a list of lists of pixels)
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    # fit polynomials of 2nd degree
    left_fit, right_fit, left_fitx, right_fitx, ploty = fit_poly(binary_warped.shape, leftx, lefty, rightx, righty)

    if visualization:
        # color in left and right pixels
        out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
        out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]
        # plot the found windows in new graph
        f, ax = plt.subplots()
        ax.imshow(out_img)
        ax.plot(left_fitx, ploty, color='yellow')
        ax.plot(right_fitx, ploty, color='yellow')
        ax.set_title('Sliding window fit')

    return left_fit, right_fit


def search_around_poly(binary_warped, left_fit, right_fit, visualization=False):
    # Choose the width of the margin around the previous polynomial to search
    margin = 100

    # Grab activated pixels
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])

    # Set search area based on supplied polynomials
    left_lane_inds = ((nonzerox > (left_fit[0] * (nonzeroy ** 2) + left_fit[1] * nonzeroy +
                                   left_fit[2] - margin)) & (nonzerox < (left_fit[0] * (nonzeroy ** 2) +
                                                                         left_fit[1] * nonzeroy + left_fit[
                                                                             2] + margin)))
    right_lane_inds = ((nonzerox > (right_fit[0] * (nonzeroy ** 2) + right_fit[1] * nonzeroy +
                                    right_fit[2] - margin)) & (nonzerox < (right_fit[0] * (nonzeroy ** 2) +
                                                                           right_fit[1] * nonzeroy + right_fit[
                                                                               2] + margin)))

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    # Fit the new polynomials
    left_fit, right_fit, left_fitx, right_fitx, ploty = fit_poly(binary_warped.shape, leftx, lefty, rightx, righty)

    if visualization:
        # Create an image to draw on and an image to show the selection window
        out_img = np.dstack((binary_warped, binary_warped, binary_warped)) * 255
        window_img = np.zeros_like(out_img)
        # Color in left and right line pixels
        out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
        out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

        # Generate a polygon to illustrate the search window area
        # And recast the x and y points into usable format for cv2.fillPoly()
        left_line_window1 = np.array([np.transpose(np.vstack([left_fitx - margin, ploty]))])
        left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx + margin,
                                                                        ploty])))])
        left_line_pts = np.hstack((left_line_window1, left_line_window2))
        right_line_window1 = np.array([np.transpose(np.vstack([right_fitx - margin, ploty]))])
        right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx + margin,
                                                                         ploty])))])
        right_line_pts = np.hstack((right_line_window1, right_line_window2))

        # Draw the lane onto the warped blank image
        cv2.fillPoly(window_img, np.int_([left_line_pts]), (0, 255, 0))
        cv2.fillPoly(window_img, np.int_([right_line_pts]), (0, 255, 0))
        result = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)

        # Create plot and draw everything
        fig, ax = plt.subplots()
        ax.imshow(result)
        ax.plot(left_fitx, ploty, color='yellow')
        ax.plot(right_fitx, ploty, color='yellow')
        ax.set_title('Search around prior fit output')

    return left_fit, right_fit


class LaneFinder(object):
    """Class to perform ego lane extraction on a birds eye image.

    Depending on it's previous findings, it will perform a search from scratch using statistics over all pixels,
    or use the previous lane estimate as a prior to find the lane.
    """

    def __init__(self, size_history=5):
        self.size_history = np.int(size_history)
        self.left_fit = None
        self.right_fit = None
        # History containers for smoothing
        self.left_fit_hist = np.zeros((size_history, 3), dtype=np.object)
        self.right_fit_hist = np.zeros((size_history, 3), dtype=np.object)
        self.left_fitx_hist = np.zeros((size_history, 3), dtype=np.object)
        self.right_fitx_hist = np.zeros((size_history, 3), dtype=np.object)

    def find_lane(self, binary_warped, visualization=False):
        # Find lane pixels
        if self.left_fit is not None and self.right_fit is not None and self.check_history():
            # If we had a good estimate in the history, use the last valid value.
            self.left_fit, self.right_fit = search_around_poly(binary_warped, self.left_fit, self.right_fit,
                                                               visualization=visualization)
        else:
            # If the history shows too many failed fits, start from scratch again.
            self.left_fit, self.right_fit = sliding_window_search(binary_warped, visualization=visualization)

        # Sanity checks and write history
        self.write_history()

        # Smoothing of current values
        self.smooth()

        # Create an image with lane visualization
        outstack = np.zeros_like(binary_warped)
        out_img = np.dstack((outstack, outstack, outstack))
        # Generate x and y values for plotting.
        ploty = np.linspace(0, binary_warped.shape[0] - 1, binary_warped.shape[0])
        left_fitx = self.left_fit[0] * ploty ** 2 + self.left_fit[1] * ploty + self.left_fit[2]
        right_fitx = self.right_fit[0] * ploty ** 2 + self.right_fit[1] * ploty + self.right_fit[2]
        lane_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
        lane_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
        lane_pts = np.hstack((lane_left, lane_right))
        cv2.fillPoly(out_img, np.int_([lane_pts]), (0, 255, 0))
        # cv2.polylines(out_img, np.int32(lane_left), True, (255, 0, 0), 15)
        # cv2.polylines(out_img, np.int32(lane_right), True, (255, 0, 0), 15)

        return out_img

    def measure_lane_geometry(self, img):
        # Define conversions in x and y from pixels space to meters
        ym_per_pix = 30 / 720  # meters per pixel in y dimension
        xm_per_pix = 3.7 / 700  # meters per pixel in x dimension

        # Define y-value where we want radius of curvature and lane center
        # We'll choose the maximum y-value, corresponding to the bottom of the image
        y_eval = np.max(img.shape[1]) * ym_per_pix
        cmax = int(img.max())

        # Calculation of curvature radii
        left_curverad = (1 + (2 * self.left_fit[0] * y_eval + self.left_fit[1]) ** 2) ** (1.5) / (
                    2 * np.abs(self.left_fit[0]))
        right_curverad = (1 + (2 * self.right_fit[0] * y_eval + self.right_fit[1]) ** 2) ** (1.5) / (
                    2 * np.abs(self.right_fit[0]))

        # Calculation of camera position to lane center
        left_fitx = self.left_fit[0] * y_eval ** 2 + self.left_fit[1] * y_eval + self.left_fit[2]
        right_fitx = self.right_fit[0] * y_eval ** 2 + self.right_fit[1] * y_eval + self.right_fit[2]
        offset_to_lane = (((left_fitx + right_fitx) / 2.) - np.max(img.shape[0])) * xm_per_pix

        # Display calculated radii on image
        color = (cmax, cmax, cmax)
        pos_left, pos_right = (50, 50), (50, 100)
        font = cv2.FONT_HERSHEY_SIMPLEX
        fontscale = 1
        lineType = 2
        if left_curverad < 10000.:
            cv2.putText(img, "radius left = {:.1f} m".format(left_curverad),
                        pos_left, font, fontscale, color, lineType)
        else:
            cv2.putText(img, "radius left = inf",
                        pos_left, font, fontscale, color, lineType)
        if right_curverad < 10000.:
            cv2.putText(img, "radius right = {:.1f} m".format(right_curverad),
                        pos_right, font, fontscale, color, lineType)
        else:
            cv2.putText(img, "radius right = inf",
                        pos_right, font, fontscale, color, lineType)
        cv2.putText(img, "camera offset = {:.1f} m".format(offset_to_lane),
                    (50, 150), font, fontscale, color, lineType)

        return img, left_curverad, right_curverad

    def write_history(self, min_lane_width=500.0, offset_b=0.5, offset_c=0.5):
        """
        Perform sanity checks on the lane fits and write to history accordingly.
        """
        a_left, a_right = self.left_fit[2], self.right_fit[2]
        b_left, b_right = self.left_fit[1], self.right_fit[1]
        c_left, c_right = self.left_fit[0], self.right_fit[0]
        confidence = 0
        # If we started from big bang add in any case, elementwise comparison due to multidim array
        if np.any(self.left_fit_hist == 0) or \
                np.any(self.left_fit_hist == 0):
            confidence += 4
        # Check curvature
        if (c_left < c_right + offset_c * np.abs(c_right)) \
                and (c_left > c_right - offset_c * np.abs(c_right)):
            confidence += 1
        # Check slope
        if (b_left < b_right + offset_b * np.abs(b_right)) \
                and (b_left > b_right - offset_b * np.abs(b_right)):
            confidence += 1
        # Check lane width in pixels
        if a_right - a_left > min_lane_width:
            confidence += 1
        else:
            confidence -= 1
        # Finally check if confidence is high enough and if so add to history,
        # else put None if there is more than one valid fit in the history
        if confidence >= 2:
            self.left_fit_hist[1:] = self.left_fit_hist[:-1]
            self.left_fit_hist[0] = self.left_fit
            self.right_fit_hist[1:] = self.right_fit_hist[:-1]
            self.right_fit_hist[0] = self.right_fit
        else:
            if np.where(self.left_fit_hist == None)[0].size < (self.size_history-1)*3:
                self.left_fit_hist[1:] = self.left_fit_hist[:-1]
                self.left_fit_hist[0] = [None, None, None]
            else:
                # if all the fits in history are unvalid we will add the result anyways.
                self.left_fit_hist[1:] = self.left_fit_hist[:-1]
                self.left_fit_hist[0] = self.left_fit
            if np.where(self.right_fit_hist == None)[0].size < (self.size_history-1)*3:
                self.right_fit_hist[1:] = self.right_fit_hist[:-1]
                self.right_fit_hist[0] = [None, None, None]
            else:
                # if all the fits in history are unvalid we will add the result anyways.
                self.right_fit_hist[1:] = self.right_fit_hist[:-1]
                self.right_fit_hist[0] = self.right_fit

    def check_history(self):
        """
        Decide if there are enough good lane finds in the history to search from pior.
        """
        if np.where(self.left_fit_hist[:3] == None)[0].size == 3*3:
            # If the last 3 iterations yielded unvalid fits, return false
            return False
        elif np.where(self.right_fit_hist[:3] == None)[0].size == 3*3:
            # If the last 3 iterations yielded unvalid fits, return false
            return False
        else:
            return True

    def smooth(self):
        """
        Performs an average over all valid fit values and stores them
        in self.left_fit and self.right_fit
        """
        valid_fits_left = self.left_fit_hist[np.logical_and(self.left_fit_hist != 0,
                                                            self.left_fit_hist != None)].reshape(-1, 3)
        valid_fits_right = self.right_fit_hist[np.logical_and(self.left_fit_hist != 0,
                                                              self.left_fit_hist != None)].reshape(-1, 3)

        self.left_fit = np.array(np.mean(valid_fits_left, axis=0), dtype=float)
        self.right_fit = np.array(np.mean(valid_fits_right, axis=0), dtype=float)
