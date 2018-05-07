from keras.models import Sequential
from keras.layers import Conv2D, Dropout, MaxPooling2D, UpSampling2D


class ColorfyModelFactory(object):
    def __init__(self, input_shape):
        self.input_shape = input_shape

    def get_model(self):
        model = Sequential()

        model.add(Conv2D(512, kernel_size=(3, 3), activation='relu', input_shape=self.input_shape, padding="same", kernel_initializer="glorot_normal"))
        model.add(Dropout(0.25))

        model.add(MaxPooling2D(pool_size=(2, 2), padding="same"))

        model.add(Conv2D(256, (5, 5), activation='relu', padding="same", kernel_initializer="glorot_normal"))
        model.add(Dropout(0.25))

        # model.add(MaxPooling2D(pool_size=(2, 2), padding="same"))
        #
        # model.add(Conv2D(128, (7, 7), activation='relu', padding="same"))
        # model.add(Dropout(0.25))
        #
        # model.add(UpSampling2D(size=(2, 2)))

        model.add(Conv2D(64, (5, 5), activation='relu', padding="same", kernel_initializer="glorot_normal"))
        model.add(Dropout(0.25))

        model.add(UpSampling2D(size=(2, 2)))

        model.add(Conv2D(2, (3, 3), activation='relu', padding="same", kernel_initializer="glorot_normal"))
        return model
