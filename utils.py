import time
import cv2
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session


def save_timestamped_result(folder, image):
    time_string = time.strftime("%Y%m%d_%H%M%S")
    cv2.imwrite('{}/results_{}.png'.format(folder, time_string), image)


def limit_tensorflow_memory_usage():
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.3
    set_session(tf.Session(config=config))
