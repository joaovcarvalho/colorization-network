import time
import cv2


def save_timestamped_result(folder, image):
    time_string = time.strftime("%Y%m%d_%H%M%S")
    cv2.imwrite('{}/results_{}.png'.format(folder, time_string), image)
