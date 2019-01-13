import cv2
import keras
import numpy as np
from PIL import ImageFile
from keras.optimizers import Adam, RMSprop

ImageFile.LOAD_TRUNCATED_IMAGES = True

from keras.callbacks import TensorBoard, ReduceLROnPlateau
import keras.backend as K
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session

config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.3
set_session(tf.Session(config=config))

from keras.preprocessing.image import ImageDataGenerator

from image_preprocessing import ColorizationDirectoryIterator
from model import ColorfyModelFactory
from weights_saver_callback import WeightsSaverCallback

TARGET_SIZE = (128, 128)

NUM_EPOCHS = 2
BATCH_SIZE = 10
STEPS_PER_EPOCH = 200000
VALIDATION_STEPS = 1000
SAVE_MODEL_EVERY_N_BATCHES = 500

model = ColorfyModelFactory(TARGET_SIZE + (1,)).get_model()
model.summary()

# weights = np.load('weights.npy')
# weights = 1 - weights
# weights_v = K.constant(weights)


def colorize_loss(y_true, y_pred):
    global weights_v
    mult = y_true - y_pred
    square = K.square(mult)
    sum = K.sum(square, axis=(1, 2))
    # weighted_sum = sum * weights_v
    return K.sum(sum)


def cross_entropy_loss(y_true, y_pred):
    mult = y_true * K.log(y_pred)
    return -1 * K.mean(mult)


optimizer = RMSprop(lr=0.0001)

model.compile(optimizer=optimizer, loss=colorize_loss)

train_datagen = ImageDataGenerator(
        rescale=1./255,
)

data_folder = 'places/data/vision/torralba/deeplearning/images256'
# data_folder = 'imagenet'

data_generator = ColorizationDirectoryIterator(
        data_folder,
        train_datagen,
        target_size=TARGET_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='original',
)

tensor_board_callback = TensorBoard(
    log_dir='./graph', histogram_freq=0, write_graph=True, write_images=True, write_grads=True
)

reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                              patience=1, min_lr=0.000001)

callbacks = [WeightsSaverCallback(model, every=SAVE_MODEL_EVERY_N_BATCHES), tensor_board_callback, reduce_lr]

model.fit_generator(
        data_generator,
        steps_per_epoch=STEPS_PER_EPOCH,
        epochs=NUM_EPOCHS,
        validation_data=data_generator,
        validation_steps=VALIDATION_STEPS,
        callbacks=callbacks,
)