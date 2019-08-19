from moviepy.editor import VideoFileClip
import numpy as np
import cv2
from camera_calibration import mtx, dist, undist_image, color_threshold, warp
from lane_finder import LaneFinder
lf = LaneFinder()


def lane_find_on_image(image):
    """ A function that takes a colored RGB image and detects and plots lanes on it

    :param image: RGB image.
    :return: RGB image with detected lanes.
    """
    img = np.copy(image)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    undist_img = undist_image(img, mtx, dist)
    binary, color_binary = color_threshold(undist_img)
    birdseye, M = warp(binary)
    lines = lf.find_lane(birdseye, visualization=False)
    output_image = cv2.addWeighted(image, 0.8, lines, 1.0, 0.0)
    return output_image


if __name__ == "__main__":
    write_output = 'output_images/project_video.mp4'
    clip1 = VideoFileClip("project_video.mp4")
    write_clip = clip1.fl_image(lane_find_on_image)
    write_clip.write_videofile(write_output, audio=False)
