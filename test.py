from preprocessing import ColorfyPreprocessing
import os
import cv2
import sys
import numpy as np
from model import ColorfyModelFactory

directory = 'data/'
files = [f for (_, _, fs) in os.walk(directory) for f in fs if f.endswith(".jpg")]

# input image dimensions
img_rows, img_cols = 128, 128
input_shape = (img_rows, img_cols)

model = ColorfyModelFactory((img_rows, img_cols, 1)).get_model()
model.load_weights(sys.argv[1])

preprocessor = ColorfyPreprocessing(directory, input_shape, cv2.COLOR_BGR2LAB)

for i in range(20):

    image = preprocessor.process(files[i])
    if image is None:
        continue

    original_bgr = cv2.cvtColor(image, cv2.COLOR_LAB2BGR)
    print(image.shape)

    gray = preprocessor.get_gray_image()

    ab_space = model.predict(np.array(gray).reshape((1, 128, 128, 1)))[0]
    a = ab_space[:, :, 0]
    b = ab_space[:, :, 1]

    a = a.reshape((128, 128, 1))
    b = b.reshape((128, 128, 1))
    gray = gray.reshape((128, 128, 1))

    colorized = np.concatenate((gray, a, b), axis=2)
    colorized = cv2.cvtColor(colorized, cv2.COLOR_LAB2BGR)

    gray = cv2.cvtColor(original_bgr, cv2.COLOR_BGR2GRAY)
    bgr_gray = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    before_after = np.concatenate((bgr_gray, colorized), axis=1)

    print(before_after.shape)

    cv2.imshow('gray', before_after)
    cv2.waitKey(0)
