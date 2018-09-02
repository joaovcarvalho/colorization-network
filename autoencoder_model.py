from keras.models import Sequential
from keras.layers import Conv2D, Dropout, UpSampling2D, AveragePooling2D, BatchNormalization
from keras.regularizers import l2

DROPOUT_RATE = 0.5
KERNEL_INITIALIZER = "glorot_normal"
CNN_ACTIVATION = 'relu'
OUTPUT_CHANNELS = 3


def add_conv_layer(depth, model, add_batch=True):
    model.add(Conv2D(
        depth,
        (5, 5),
        activation=CNN_ACTIVATION,
        padding="same",
        kernel_initializer=KERNEL_INITIALIZER,
        kernel_regularizer=l2(0.)
    ))
    if add_batch:
        model.add(BatchNormalization())


def add_conv_layers(how_many, model):
    min_convolutions = 32
    max_covolutions = 512
    for i in range(how_many):
        iteration = how_many - 1 - i
        depth = max(min_convolutions, min(max_covolutions, 2**iteration))
        model.add(Conv2D(
            depth,
            (3, 3),
            activation=CNN_ACTIVATION,
            padding="same",
            kernel_initializer=KERNEL_INITIALIZER,
            kernel_regularizer=l2(0.)
            ))
        model.add(BatchNormalization())


class AutoEncoderFactory(object):
    def __init__(self, input_shape):
        self.input_shape = input_shape

    def get_model(self):
        model = Sequential()

        # Input Layer
        model.add(Conv2D(64,
                         kernel_size=(3, 3),
                         activation=CNN_ACTIVATION,
                         input_shape=self.input_shape,
                         padding="same",
                         kernel_initializer=KERNEL_INITIALIZER))

        # Hidden Layers
        # add_conv_layers(10, model)

        add_conv_layer(512, model)
        model.add(AveragePooling2D())
        add_conv_layer(256, model, add_batch=False)
        add_conv_layer(256, model)
        add_conv_layer(128, model)
        add_conv_layer(64, model)
        model.add(AveragePooling2D())
        add_conv_layer(32, model, add_batch=False)
        add_conv_layer(32, model)
        add_conv_layer(32, model)
        model.add(UpSampling2D())
        add_conv_layer(32, model)
        model.add(UpSampling2D())
        add_conv_layer(16, model)

        # Output layer
        model.add(Conv2D(OUTPUT_CHANNELS, (3, 3), activation=CNN_ACTIVATION, padding="same", kernel_initializer=KERNEL_INITIALIZER))

        return model
