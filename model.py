from keras.models import Sequential
from keras.layers import Conv2D, Dropout, UpSampling2D, AveragePooling2D
from keras.regularizers import l2

DROPOUT_RATE = 0.5
KERNEL_INITIALIZER = "glorot_normal"
CNN_ACTIVATION = 'relu'

def add_conv_layers(how_many, model):
    min_convolutions = 32
    max_covolutions = 512
    for i in range(how_many):
        iteration = how_many - 1 - i
        depth = max(min_convolutions, min(max_covolutions, 2**iteration))
        model.add(Conv2D(depth, (3, 3), 
            activation=CNN_ACTIVATION, 
            padding="same", 
            kernel_initializer=KERNEL_INITIALIZER, 
            kernel_regularizer=l2(0.)
            ))

class ColorfyModelFactory(object):
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
        add_conv_layers(10, model)

        # Output layer
        model.add(Conv2D(3, (3, 3), activation=CNN_ACTIVATION, padding="same", kernel_initializer=KERNEL_INITIALIZER))

        return model
