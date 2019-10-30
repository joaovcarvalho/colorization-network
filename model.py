from keras import Model, Input
from keras.layers import Conv2D, Dropout, UpSampling2D, AveragePooling2D, BatchNormalization, Activation, Softmax, \
    Reshape
from keras.layers.core import Activation
from keras.regularizers import l2

DROPOUT_RATE = 0.5
KERNEL_INITIALIZER = "glorot_normal"
CNN_ACTIVATION = "relu"
OUTPUT_CHANNELS = 256


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

        x = add_conv_layer(64, x)
        x = add_conv_layer(64, x, add_batch=True, strides=2)
        x = add_conv_layer(128, x)
        x = add_conv_layer(128, x)
        x = add_conv_layer(128, x, add_batch=True, strides=2)
        x = add_conv_layer(256, x)
        x = add_conv_layer(256, x)
        x = add_conv_layer(256, x)
        x = add_conv_layer(256, x, add_batch=True, strides=2)
        x = add_conv_layer(512, x)
        x = add_conv_layer(512, x)
        x = add_conv_layer(512, x)
        x = add_conv_layer(512, x, add_batch=True)
        x = add_conv_layer(512, x)
        x = add_conv_layer(512, x)
        x = add_conv_layer(512, x)
        x = add_conv_layer(512, x, add_batch=True)
        x = UpSampling2D()(x)
        x = add_conv_layer(256, x)
        x = add_conv_layer(256, x)
        x = add_conv_layer(256, x, add_batch=True)
        x = UpSampling2D()(x)
        x = add_conv_layer(128, x)
        x = add_conv_layer(128, x)
        x = UpSampling2D()(x)
        x = add_conv_layer(64, x)
        x = add_conv_layer(64, x)
        x = add_conv_layer(64, x)

        # Output layer
        x = Conv2D(OUTPUT_CHANNELS, (1, 1),
                   padding="same",
                   kernel_initializer=KERNEL_INITIALIZER)(x)

        # shape = x.output_shape
        # shape_for_softmax = (shape[1] * shape[2], shape[3])
        # x = Reshape(shape_for_softmax)(x)

        x = Softmax(axis=-1)(x)

        # x = Reshape(shape)(x)

        model = Model(inputs=net_input, outputs=x)

        return model
