from keras import Model, Input
from keras.models import Sequential
from keras.layers import Conv2D, Dropout, UpSampling2D, AveragePooling2D, BatchNormalization
from keras.regularizers import l2
from keras.applications import VGG19

DROPOUT_RATE = 0.5
KERNEL_INITIALIZER = "glorot_normal"
CNN_ACTIVATION = 'relu'
OUTPUT_CHANNELS = 144


def add_conv_layer(depth, x, add_batch=True):
    x = Conv2D(
        depth,
        (5, 5),
        activation=CNN_ACTIVATION,
        padding="same",
        kernel_initializer=KERNEL_INITIALIZER,
        kernel_regularizer=l2(0.01)
    )(x)
    if add_batch:
        x = BatchNormalization()(x)
    return x


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


class ColorfyModelFactory(object):
    def __init__(self, input_shape):
        self.input_shape = input_shape

    def get_model(self):
        net_input = Input(shape=self.input_shape, name='net_input')

        # Input Layer
        x = Conv2D(256,
                   kernel_size=(3, 3),
                   activation=CNN_ACTIVATION,
                   padding="same",
                   kernel_initializer=KERNEL_INITIALIZER)(net_input)

        x = add_conv_layer(256, x, add_batch=False)
        x = add_conv_layer(128, x, )
        x = add_conv_layer(64, x, )
        x = add_conv_layer(32, x, add_batch=False)
        x = add_conv_layer(32, x, )
        x = add_conv_layer(16, x, )

        # Output layer
        x = Conv2D(OUTPUT_CHANNELS, (3, 3), activation=CNN_ACTIVATION, padding="same", kernel_initializer=KERNEL_INITIALIZER)(x)

        model = Model(inputs=net_input, outputs=x)

        return model