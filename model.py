from keras.models import Sequential
from keras.layers import Conv2D, Dropout, UpSampling2D, AveragePooling2D

DROPOUT_RATE = 0.5
KERNEL_INITIALIZER = "glorot_normal"
CNN_ACTIVATION = 'relu'


class ColorfyModelFactory(object):
    def __init__(self, input_shape):
        self.input_shape = input_shape

    def get_model(self):
        model = Sequential()

        model.add(Conv2D(64,
                         kernel_size=(3, 3),
                         activation=CNN_ACTIVATION,
                         input_shape=self.input_shape,
                         padding="same",
                         kernel_initializer=KERNEL_INITIALIZER))

        model.add(Conv2D(128, (3, 3), activation=CNN_ACTIVATION, padding="same", kernel_initializer=KERNEL_INITIALIZER))

        model.add(Conv2D(256, (3, 3), activation=CNN_ACTIVATION, padding="same"))

        model.add(Conv2D(128, (3, 3), activation=CNN_ACTIVATION, padding="same", kernel_initializer=KERNEL_INITIALIZER))

        model.add(Conv2D(64, (3, 3), activation=CNN_ACTIVATION, padding="same", kernel_initializer=KERNEL_INITIALIZER))

        model.add(Conv2D(3, (3, 3), activation=CNN_ACTIVATION, padding="same", kernel_initializer=KERNEL_INITIALIZER))

        return model
