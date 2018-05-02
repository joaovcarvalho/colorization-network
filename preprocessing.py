import StringIO

import cv2
from cv2 import imread, resize
import numpy as np
from PIL import Image


class ColorfyPreprocessing(object):

    def __init__(self, directory, input_size, color_space):
        self.directory = directory
        self.input_size = input_size
        self.color_space = color_space
        self.current_image = None

    def load_image(self, file_name):
        with open(self.directory + file_name, 'rb') as img_bin:
            buff = StringIO.StringIO()
            buff.write(img_bin.read())
            buff.seek(0)

            pil_image = Image.open(buff)
            temp_img = np.asarray(pil_image, dtype=np.uint8)
            image_size = temp_img.shape
            if len(image_size) == 3:
                image = cv2.cvtColor(temp_img, cv2.COLOR_RGB2BGR)
            else:
                return None

        if image is not None:
            return image
        else:
            return None

    def convert_to_lab(self, image):
        return cv2.cvtColor(image, self.color_space)

    def resize_image(self, image):
        return resize(image, self.input_size)

    def process(self, file_name):
        loaded_image = self.load_image(file_name)
        if loaded_image is not None:
            self.current_image = self.resize_image(self.convert_to_lab(loaded_image))
            return self.current_image
        else:
            return None

    def get_gray_image(self):
        l_channel, a_channel, b_channel = cv2.split(self.current_image)
        l_channel /= 100
        return l_channel

    def get_color_image(self):
        l_channel, a_channel, b_channel = cv2.split(self.current_image)
        a_channel /= 127
        b_channel /= 127
        return cv2.merge([a_channel, b_channel])