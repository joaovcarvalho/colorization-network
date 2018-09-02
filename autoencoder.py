from keras.datasets import cifar10
from weights_saver_callback import WeightsSaverCallback
from keras.callbacks import TensorBoard
from autoencoder_model import AutoEncoderFactory
import numpy as np

# input image dimensions
img_rows, img_cols = 32, 32
input_shape = (img_rows, img_cols)

model = AutoEncoderFactory((img_rows, img_cols, 3)).get_model()

# For a mean squared error regression problem
model.compile(optimizer='adadelta', loss='mse')

(x_train, _), (x_test, _) = cifar10.load_data()

train_size = x_train.shape[0]
test_size = x_test.shape[0]

y_train = x_train
y_test = x_test

x_train = np.array(x_train).reshape(train_size, img_rows, img_cols, 3)
x_test = np.array(x_test).reshape(test_size, img_rows, img_cols, 3)

NUM_EPOCHS = 10
BATCH_SIZE = 32
SAVE_MODEL_EVERY_N_BATCHES = 10

tbCallBack = TensorBoard(log_dir='./graph',
                         batch_size=BATCH_SIZE,
                         histogram_freq=2,
                         write_graph=True,
                         write_images=True,
                         write_grads=True)

callbacks = [WeightsSaverCallback(model, every=SAVE_MODEL_EVERY_N_BATCHES), tbCallBack]

input_images = x_train.astype(float)
output_images = x_train.astype(float)

input_images /= 255
output_images /= 255

input_images -= .5
output_images -= .5

history = model.fit(input_images, output_images,
                    validation_split=0.1,
                    epochs=NUM_EPOCHS,
                    callbacks=callbacks,
                    batch_size=BATCH_SIZE)
