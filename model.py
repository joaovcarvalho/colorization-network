from keras import Model, Input
from keras.layers import Conv2D, Dropout, UpSampling2D, AveragePooling2D, BatchNormalization, Activation, Softmax, \
    Concatenate
from keras.regularizers import l2

DROPOUT_RATE = 0.5
KERNEL_INITIALIZER = "glorot_normal"
CNN_ACTIVATION = 'relu'
OUTPUT_CHANNELS = 256


def add_conv_layer(depth, x, add_batch=False, strides=1, dilation_rate=1, kernel_size=3):
    x = Conv2D(
        depth,
        (kernel_size, kernel_size),
        activation=CNN_ACTIVATION,
        padding="same",
        strides=strides,
        dilation_rate=dilation_rate,
        kernel_initializer=KERNEL_INITIALIZER,
        # Change l2 regularization to 0.001 to be equal to ColorfulColorization model
        kernel_regularizer=l2(0.03)
    )(x)

    if add_batch:
        x = BatchNormalization()(x)
    return x


class ColorfyModelFactory(object):
    def __init__(self, input_shape):
        self.input_shape = input_shape

    def get_model(self):
        net_input = Input(shape=self.input_shape, name='net_input')

        # Conv 1
        x = Conv2D(64,
                   kernel_size=(3, 3),
                   activation=CNN_ACTIVATION,
                   padding="same",
                   kernel_initializer=KERNEL_INITIALIZER)(net_input)

        x = add_conv_layer(64, x, add_batch=True, strides=2)

        # Conv 2
        x = add_conv_layer(128, x, add_batch=True, strides=2)

        # Conv 3
        x = add_conv_layer(256, x)
        x = add_conv_layer(256, x, add_batch=True, strides=2)

        # Conv 4
        x = add_conv_layer(512, x)
        x = add_conv_layer(512, x)
        x = add_conv_layer(512, x, add_batch=True)

        # Conv 5
        x = add_conv_layer(512, x, dilation_rate=2)
        x = add_conv_layer(512, x, dilation_rate=2)
        x = add_conv_layer(512, x, add_batch=True, dilation_rate=2)

        # Conv 6
        x = UpSampling2D()(x)
        x = add_conv_layer(512, x, dilation_rate=2)
        x = add_conv_layer(512, x, dilation_rate=2)
        x = add_conv_layer(512, x, add_batch=True, dilation_rate=2)

        # Conv 7
        x = UpSampling2D()(x)
        x = add_conv_layer(256, x)
        x = add_conv_layer(256, x)
        x = add_conv_layer(256, x, add_batch=True)

        # Conv 8
        x = UpSampling2D()(x)
        x = add_conv_layer(128, x)
        x = add_conv_layer(128, x)

        # Output layer
        x = Conv2D(OUTPUT_CHANNELS, (1, 1),
                   padding="same",
                   kernel_initializer=KERNEL_INITIALIZER)(x)

        x = Softmax(axis=(1, 2))(x)

        model = Model(inputs=net_input, outputs=x)

        return model
