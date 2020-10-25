import base64
from io import BytesIO
from os import path

import Algorithmia
import io
from keras.preprocessing.image import ImageDataGenerator
from image_preprocessing import ColorizationDirectoryIterator
from PIL import Image

from post_processing import get_image_from_network_result

API_KEY = 'sim3UgzRdcjDo3ouAIRFalFDB/D1'
COLORFUL_ALGO_PATH = 'deeplearning/ColorfulImageColorization/1.0.1'
example_image = "data:image/jpeg;base64,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"

image_path = './colorful/example.png'

def download_colorful_image(base64_image, path_to_save):
    input = {"image": base64_image}
    client = Algorithmia.client(API_KEY)
    algo = client.algo(COLORFUL_ALGO_PATH)

    output = algo.pipe(input)
    result = output.result

    output_path = result['output']

    if client.file(output_path).exists():
        bytes = client.file(output_path).getBytes()
        image = Image.open(io.BytesIO(bytes))
        image = image.convert('RGB')
        image.save(path_to_save)


if __name__ == "__main__":
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

        original, _ = get_image_from_network_result(
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

        if not path.exists(image_path):
            print('Download file {}'.format(image_path))
            download_colorful_image(bas64, image_path)
