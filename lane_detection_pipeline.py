from moviepy.editor import VideoFileClip
import numpy as np
import cv2
from camera_calibration import cal_cam, undist_image, color_threshold, warp_to_birdseye, warp
from lane_finder import LaneFinder

# Calibrate the camera
mtx, dist = cal_cam()
# Create Lanefinder instance
lf = LaneFinder()

def lane_find_on_image(image):
    """
    A function that takes a colored RGB image and detects and plots lanes on it
    """
    img = np.copy(image)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    undist_img = undist_image(img, mtx, dist)
    binary, color_binary = color_threshold(undist_img)
    birdseye, M, Minv = warp_to_birdseye(binary)
    lines = lf.find_lane(birdseye)
    lines = warp(lines, Minv)
    lines, left_rad, right_rad = lf.measure_lane_geometry(lines)
    output_image = cv2.addWeighted(image, 1.0, lines, 0.3, 0.0)
    return output_image


if __name__ == "__main__":
    write_output = 'output_images/project_video.mp4'
    clip1 = VideoFileClip("project_video.mp4")
    write_clip = clip1.fl_image(lane_find_on_image)
    write_clip.write_videofile(write_output, audio=False)
