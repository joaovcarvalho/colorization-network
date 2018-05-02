from keras.models import Sequential
from keras.layers import Conv2D, Dropout


class ColorfyModelFactory(object):
    def __init__(self, input_shape):
        self.input_shape = input_shape

    def get_model(self):
        model = Sequential()

        model.add(Conv2D(512, kernel_size=(3, 3), activation='relu', input_shape=self.input_shape, padding="same"))
        model.add(Dropout(0.25))

        model.add(Conv2D(256, (3, 3), activation='relu', padding="same"))
        model.add(Dropout(0.25))

        model.add(Conv2D(128, (3, 3), activation='relu', padding="same"))
        model.add(Dropout(0.25))

        model.add(Conv2D(64, (3, 3), activation='relu', padding="same"))
        model.add(Dropout(0.25))

        model.add(Conv2D(2, (3, 3), activation='relu', padding="same"))
        return model
