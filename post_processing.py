import cv2
import numpy as np
from quantization import convert_quantization_to_image, convert_quantization_to_image_average


def get_image_from_network_result(
        result,
        l_channel,
        size_tuple,
        OUTPUT_SIZE,
        use_average=True
):
    (img_rows, img_cols) = size_tuple
    if use_average:
        color_space = convert_quantization_to_image_average(
                result, 16, 256, 256
        )
    else:
        color_space = convert_quantization_to_image(result, 16, 256)

    a = color_space[:, :, 0]
    b = color_space[:, :, 1]

    a = a.reshape((img_rows, img_cols, 1))
    b = b.reshape((img_rows, img_cols, 1))

    lab_original = np.concatenate((l_channel, a, b), axis=2).astype('uint8')
    resized_image = cv2.resize(lab_original, OUTPUT_SIZE)
    colorized = cv2.cvtColor(resized_image, cv2.COLOR_LAB2BGR)
    return colorized, lab_original