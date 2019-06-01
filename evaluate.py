import sys

import keras.backend as K
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator

from image_preprocessing import ColorizationDirectoryIterator
from model import ColorfyModelFactory

img_rows = 64
img_cols = 64

input_shape = (img_rows, img_cols)

model = ColorfyModelFactory(input_shape + (1,)).get_model()
model.load_weights(sys.argv[1])


def colorize_loss(y_true, y_pred):
    global weights_v
    mult = y_true - y_pred
    square = K.square(mult)
    sum = K.sum(square, axis=(1, 2))
    return K.sum(sum)


def categorical_accuracy(y_true, y_pred):
    correct_indices = y_true * y_pred
    # accuracy = K.sum(correct_indices, axis=0) / K.sum(y_true, axis=0)
    accuracy = K.sum(correct_indices, axis=0)
    return K.cast(K.mean(accuracy), K.floatx())


def auroc(y_true, y_pred):
    auc = K.tf.metrics.auc(y_true, y_pred)[1]
    K.get_session().run(K.tf.local_variables_initializer())
    return auc


train_datagen = ImageDataGenerator(
    rescale=1. / 255,
)

train_generator = ColorizationDirectoryIterator(
    'imagenet',
    train_datagen,
    target_size=input_shape,
    batch_size=1,
    class_mode='original'
)

how_many = min(len(train_generator) - 1, 1000)
model.compile(optimizer=Adam(), loss=colorize_loss, metrics=[categorical_accuracy, auroc])
loss, accuracy, auroc = model.evaluate_generator(train_generator, steps=how_many)
print('Accuracy {}'.format(accuracy * 100))
print('AUC ROC {}'.format(auroc * 100))

