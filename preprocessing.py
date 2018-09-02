import cv2
from cv2 import resize


class ColorfyPreprocessing(object):

    def __init__(self, input_size, color_space):
        self.input_size = input_size
        self.color_space = color_space
        self.current_image = None

    def convert_to_lab(self, image):
        return cv2.cvtColor(image, self.color_space)

    def resize_image(self, image):
        return resize(image, self.input_size)

    def process(self, image):
        if image is not None:
            self.current_image = self.resize_image(self.convert_to_lab(image))
            return self.current_image
        else:
            return None

    def get_gray_image(self, image):
        l_channel, a_channel, b_channel = cv2.split(image)
        return l_channel

    def get_color_image(self, image):
        l_channel, a_channel, b_channel = cv2.split(image)
        return cv2.merge([a_channel, b_channel])
