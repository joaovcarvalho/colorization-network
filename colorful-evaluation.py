import base64
from io import BytesIO
from os import path
import numpy as np

from keras.preprocessing.image import ImageDataGenerator

from compare import compare_lab_images
from image_preprocessing import ColorizationDirectoryIterator
from PIL import Image
import cv2

from post_processing import get_image_from_network_result

accuracies = []

img_rows = 128
img_cols = 128

input_shape = (img_rows, img_cols)

train_datagen = ImageDataGenerator(
    rescale=1. / 255,
)

train_generator = ColorizationDirectoryIterator(
    '../places-dataset',
    train_datagen,
    target_size=input_shape,
    batch_size=1,
    class_mode='original',
    seed=3,
)

count = 0
HOW_MANY_IMAGES = 1000

for input, output in train_generator:
    if count >= HOW_MANY_IMAGES:
        break
    count += 1

    l_channel = input.reshape(img_rows, img_cols, 1)
    l_channel *= 7.61
    l_channel += 119.85
    l_channel = l_channel.astype('uint8')

    original, lab_original = get_image_from_network_result(
        output[0],
        l_channel,
        (img_rows, img_cols),
        (img_rows, img_cols),
        use_average=False
    )

    img = Image.fromarray(original)
    buffer = BytesIO()
    img.save(buffer, format="JPEG")
    myimage = buffer.getvalue()
    bas64 = "data:image/jpeg;base64," + base64.b64encode(myimage)

    image_path = './colorful/{}.jpg'.format(count)

    if path.exists(image_path):
        image = cv2.imread(image_path)
        lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        accuracy = compare_lab_images(lab_image, lab_original, 25) * 100

        cumulative_accuracy = [
            compare_lab_images(lab_image, lab_original, i) * 100
            for i in range(150)
        ]

        print('#{}'.format(count))

        area_under_curve = sum(cumulative_accuracy) / (150 * 100)
        print('AUC: ', area_under_curve)

        accuracies.append(sum(cumulative_accuracy))

results = np.array(accuracies)
auc_final = sum(accuracies) / (len(accuracies) * 150 * 100)
print('AUC: {:.2f}'.format(auc_final * 100))
print('Accuracy: {}'.format(np.average(results)))
print('Std: {}'.format(np.std(results)))