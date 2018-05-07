import StringIO

import cv2
from cv2 import resize
import numpy as np
from PIL import Image


class ColorfyImageLoader(object):

    def __init__(self, directory):
        self.directory = directory
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
