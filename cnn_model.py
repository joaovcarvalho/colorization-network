from image_path_sequence import ImagesPathSequence
from weights_saver_callback import WeightsSaverCallback
from model import ColorfyModelFactory
from preprocessing import ColorfyPreprocessing
import os
import cv2

directory = 'data/'
files = [f for (_, _, fs) in os.walk(directory) for f in fs if f.endswith(".jpg")]

# input image dimensions
img_rows, img_cols = 32, 32
input_shape = (img_rows, img_cols)

preprocessor = ColorfyPreprocessing(directory, input_shape, cv2.COLOR_BGR2LAB)

model = ColorfyModelFactory((img_rows, img_cols, 1)).get_model()

# For a mean squared error regression problem
model.compile(optimizer='adam', loss='mse')

history = model.fit_generator(
    ImagesPathSequence(files, 16, preprocessor),
    callbacks=[WeightsSaverCallback(model, every=10)],
    epochs=1
)

# summarize history for loss
# plt.plot(history.history['loss'])
# plt.plot(history.history['val_loss'])
# plt.title('model loss')
# plt.ylabel('loss')
# plt.xlabel('epoch')
# plt.legend(['train', 'test'], loc='upper left')
# plt.savefig('learning_curve.png')
#
# model.save("colorfy")
