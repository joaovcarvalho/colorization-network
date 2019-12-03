from keras import Model, Input
from keras.layers import Conv2D, Dropout, UpSampling2D, AveragePooling2D, BatchNormalization, Activation, Softmax, \
    Reshape, MaxPooling2D, concatenate
from keras.layers.core import Activation
from keras.regularizers import l2

DROPOUT_RATE = 0.5
KERNEL_INITIALIZER = "glorot_normal"
CNN_ACTIVATION = "relu"
OUTPUT_CHANNELS = 256


def add_inception_layer(depth, layer_input):
    tower_1 = Conv2D(depth, (1, 1), padding='same', activation=CNN_ACTIVATION)(layer_input)
    tower_1 = Conv2D(depth, (3, 3), padding='same', activation=CNN_ACTIVATION)(tower_1)

    tower_2 = Conv2D(depth, (1, 1), padding='same', activation=CNN_ACTIVATION)(layer_input)
    tower_2 = Conv2D(depth, (3, 3), padding='same', activation=CNN_ACTIVATION)(tower_2)
    tower_2 = Conv2D(depth, (3, 3), padding='same', activation=CNN_ACTIVATION)(tower_2)

    tower_3 = MaxPooling2D((3, 3), strides=(1, 1), padding='same')(layer_input)
    tower_3 = Conv2D(depth, (1, 1), padding='same', activation=CNN_ACTIVATION)(tower_3)

    output = concatenate([tower_1, tower_2, tower_3], axis=3)
    return output


def add_conv_layer(depth, x, add_batch=False, strides=1, dilation_rate=1, kernel_size=3):
    x = Conv2D(
        depth,
        (kernel_size, kernel_size),
        padding="same",
        strides=strides,
        dilation_rate=dilation_rate,
        kernel_initializer=KERNEL_INITIALIZER,
        kernel_regularizer=l2(0.01)
    )(x)

    if add_batch:
        x = BatchNormalization()(x)

    x = Activation(CNN_ACTIVATION)(x)
    return x


class ColorfyModelFactory(object):
    def __init__(self, input_shape):
        self.input_shape = input_shape

    def get_model(self):
        net_input = Input(shape=self.input_shape, name="net_input")

        # Input Layer
        x = Conv2D(64,
                   kernel_size=(3, 3),
                   activation=CNN_ACTIVATION,
                   padding="same",
                   kernel_initializer=KERNEL_INITIALIZER)(net_input)

        original_size = add_conv_layer(64, x)
        x = add_conv_layer(64, original_size, add_batch=True, strides=2)
        x = add_conv_layer(128, x)
        half_size = add_conv_layer(128, x)
        x = add_conv_layer(128, half_size, add_batch=True, strides=2)
        x = add_inception_layer(64, x)
        x = add_inception_layer(64, x)
        quarter_size = add_inception_layer(64, x)
        x = add_conv_layer(256, quarter_size, add_batch=True, strides=2)
        x = add_inception_layer(64, x)
        x = add_inception_layer(64, x)
        x = add_inception_layer(64, x)
        x = add_conv_layer(512, x, add_batch=True)
        x = add_inception_layer(128, x)
        x = add_inception_layer(128, x)
        x = add_inception_layer(128, x)
        x = add_conv_layer(512, x, add_batch=True)
        x = UpSampling2D()(x)
        x = concatenate([x, quarter_size], axis=3)
        x = add_conv_layer(256, x)
        x = add_conv_layer(256, x)
        x = add_conv_layer(256, x, add_batch=True)
        x = UpSampling2D()(x)
        x = concatenate([x, half_size], axis=3)
        x = add_conv_layer(128, x)
        x = add_conv_layer(128, x)
        x = UpSampling2D()(x)
        x = concatenate([x, original_size], axis=3)
        x = add_conv_layer(64, x)
        x = add_conv_layer(64, x)
        x = add_conv_layer(64, x)

        # Output layer
        x = Conv2D(OUTPUT_CHANNELS, (1, 1),
                   padding="same",
                   kernel_initializer=KERNEL_INITIALIZER)(x)

        x = Softmax(axis=-1)(x)

        model = Model(inputs=net_input, outputs=x)

        return model
