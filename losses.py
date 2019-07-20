from keras import backend as K


def colorize_loss(y_true, y_pred):
    difference = y_true - y_pred
    square = K.square(difference)
    # sum = K.sum(square, axis=0)
    return K.sum(square)


def colorful_colorization_loss(y_true, y_pred):
    return -1 * K.sum(y_true * K.log(y_pred + 1e-10))
