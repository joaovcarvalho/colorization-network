import cv2
from cv2 import imread, resize


class ColorfyPreprocessing(object):

    def __init__(self, directory, input_size, color_space):
        self.directory = directory
        self.input_size = input_size
        self.color_space = color_space
        self.current_image = None

    def load_image(self, file_name):
        image = imread(self.directory + file_name)
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
        return l_channel

    def get_color_image(self):
        l_channel, a_channel, b_channel = cv2.split(self.current_image)
        return cv2.merge([a_channel, b_channel])