from preprocessing import ColorfyPreprocessing
import cv2
import os

directory = 'data/'

input_size = (128, 128)

preprocessor = ColorfyPreprocessing(directory, input_size, cv2.COLOR_BGR2LAB)

files = [f for (_, _, fs) in os.walk(directory) for f in fs if f.endswith(".jpg")]

for file_name in files:
    try:
        image = preprocessor.process(file_name)
        if image is None:
            os.remove(directory + file_name)
            print("Failed image - {} deleted".format(file_name))
    except Exception as e:
        os.remove(directory + file_name)
        print("Failed image - {} deleted".format(file_name))
        print(e)
