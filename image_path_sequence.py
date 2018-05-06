import numpy as np
from keras.utils import Sequence


class ImagesPathSequence(Sequence):

    def __init__(self, x_set, batch_size, preprocessor):
        self.x = x_set
        self.batch_size = batch_size
        self.preprocessor = preprocessor

    def __len__(self):
        return int(np.ceil(len(self.x) / float(self.batch_size)))

    def __getitem__(self, idx):
        try:
            images = []
            gray_images = []
            color_images = []

            i = 0
            while len(images) < self.batch_size:
                try:
                    next_file_path = self.x[idx*self.batch_size + i]
                except IndexError:
                    break

                image = self.preprocessor.process(next_file_path)
                if image is not None:
                    images.append(image)
                    gray_images.append(self.preprocessor.get_gray_image())
                    color_images.append(self.preprocessor.get_color_image())
                i += 1

            image_rows, image_cols = self.preprocessor.input_size
            gray_images, color_images = np.array(gray_images).reshape((self.batch_size, image_rows, image_cols, 1)), \
                                        np.array(color_images).reshape((self.batch_size, image_rows, image_cols, 2))
            return gray_images, color_images
        except Exception as e:
            print(e)