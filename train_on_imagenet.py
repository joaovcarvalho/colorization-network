import sys

import tensorflow as tf
from PIL import ImageFile
from keras.backend.tensorflow_backend import set_session
from keras.callbacks import TensorBoard, ReduceLROnPlateau
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator

from image_preprocessing import ColorizationDirectoryIterator
from losses import colorful_colorization_loss
from model import ColorfyModelFactory
from weights_saver_callback import WeightsSaverCallback

ImageFile.LOAD_TRUNCATED_IMAGES = True

config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.5
set_session(tf.Session(config=config))

TARGET_SIZE = (64, 64)

HOW_MANY_IMAGES_PER_EPOCH = 2.3e6
NUM_EPOCHS = 4
BATCH_SIZE = 20
STEPS_PER_EPOCH = HOW_MANY_IMAGES_PER_EPOCH / BATCH_SIZE
SAVE_MODEL_EVERY_N_BATCHES = 500

model = ColorfyModelFactory(TARGET_SIZE + (1,)).get_model()

if len(sys.argv) > 1:
    model.load_weights(sys.argv[1])

model.summary()

# Parameters extracted from Colorful Image Colorization paper
initial_learning_rate = 3 / 10e3
optimizer = Adam(lr=initial_learning_rate, beta_1=0.9, beta_2=0.99, decay=1/10e3)

model.compile(optimizer=optimizer, loss=colorful_colorization_loss)

train_datagen = ImageDataGenerator(
    rescale=1. / 255,
)

data_folder = 'places/data/vision/torralba/deeplearning/images256'

data_generator = ColorizationDirectoryIterator(
    data_folder,
    train_datagen,
    target_size=TARGET_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='original',
)

VALIDATION_STEPS = 1000
VALIDATION_BATCH_SIZE = 10

validation_generator = ColorizationDirectoryIterator(
    data_folder,
    train_datagen,
    target_size=TARGET_SIZE,
    batch_size=VALIDATION_BATCH_SIZE,
    class_mode='original',
)

tensor_board_callback = TensorBoard(
    log_dir='./graph', histogram_freq=0, write_graph=True, write_images=True, write_grads=True
)

reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                              patience=1)

callbacks = [WeightsSaverCallback(model, every=SAVE_MODEL_EVERY_N_BATCHES), tensor_board_callback]

model.fit_generator(
    data_generator,
    steps_per_epoch=STEPS_PER_EPOCH,
    epochs=NUM_EPOCHS,
    validation_data=validation_generator,
    validation_steps=VALIDATION_STEPS,
    callbacks=callbacks,
)
