import cv2
import matplotlib.pyplot as plt

import numpy as np

image_shape = (256, 256, 3)
font = cv2.FONT_HERSHEY_SIMPLEX


def create_image_for_luminance(l):
    image = np.zeros(image_shape, dtype="float32")

    image[:, :, 0] = l
    for i in range(256):
        image[i, :, 1] = i - 127

    for j in range(256):
        image[:, j, 2] = j - 127
    return image


L = [0, 25, 50, 75, 100]
final_image_shape = (image_shape[0] * len(L), image_shape[1], image_shape[2])

images = []
for l in L:
    image = create_image_for_luminance(l)
    cv2.putText(image, 'L = {}'.format(l), (50, 50), font, 1, (100, 0, 0), 2, cv2.LINE_AA)
    images.append(image)

final_image = np.concatenate(images, axis=1).astype('float32')
colorized = cv2.cvtColor(final_image, cv2.COLOR_LAB2BGR)
# OUTPUT_SIZE = (800, 800)
# colorized = cv2.resize(colorized, OUTPUT_SIZE)
# cv2.imshow('test', colorized)
# cv2.imwrite('lab_space_color.jpg', colorized)
plt.imshow(colorized)
plt.show()
# cv2.waitKey(0)
