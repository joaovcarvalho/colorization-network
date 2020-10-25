import sys
import matplotlib.pyplot as plt

from keras.preprocessing.image import ImageDataGenerator

from compare import compare_lab_images
from image_preprocessing import ColorizationDirectoryIterator
from model import ColorfyModelFactory
from post_processing import get_image_from_network_result
import numpy as np

img_rows = 128
img_cols = 128

input_shape = (img_rows, img_cols)

model = ColorfyModelFactory(input_shape + (1,)).get_model()
model.load_weights(sys.argv[1])

train_datagen = ImageDataGenerator(rescale=1./255)

HOW_MANY_IMAGES = 1000

def plot_graph(data):
    plt.subplot(2, 1, 2)
    x = range(len(data))
    plt.ylim(0, 100)
    plt.plot(x, data)
    plt.xlabel('Limite(t)')
    plt.ylabel('Acuracia(%)')


train_generator = ColorizationDirectoryIterator(
        '../places-dataset',
        train_datagen,
        target_size=input_shape,
        batch_size=1,
        class_mode='original',
        seed=3,
    )

final_accuracies = np.zeros((HOW_MANY_IMAGES, 256))
count = 0

for input, output in train_generator:
    print('Image: #{}'.format(count))

    batch_result = model.predict(input)
    l_channel = input.reshape(img_rows, img_cols, 1)

    l_channel *= 7.61
    l_channel += 119.85
    l_channel = l_channel.astype('uint8')

    prediction = batch_result[0]

    colorized, lab_image = get_image_from_network_result(
        prediction,
        l_channel,
        (img_rows, img_cols),
        (img_rows, img_cols),
    )
    original, lab_original = get_image_from_network_result(
        output[0],
        l_channel,
        (img_rows, img_cols),
        (img_rows, img_cols),
        use_average=False
    )

    for current_accuracy in range(256):
        accuracy = compare_lab_images(lab_image, lab_original, current_accuracy)
        final_accuracies[count][current_accuracy] = accuracy

    count += 1
    if count >= HOW_MANY_IMAGES:
        break

print('Saving results...')
np.save('accuracies.npy', final_accuracies)
print('Finished')
# averages = np.average(final_accuracies, axis=0)
# print(averages.shape)
# plot_graph(averages)
# plt.show()

