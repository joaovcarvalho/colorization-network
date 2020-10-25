import numpy as np

def reformat_image(lab_image):
    image = lab_image[:, :, 1:]
    shape = image.shape
    new_shape = (shape[0] * shape[1], shape[2])
    reshaped_image = np.reshape(image, new_shape)
    return reshaped_image


def compare_lab_images(first_image, second_image, threshold=20):
    reshaped_first = reformat_image(first_image)
    reshaped_second = reformat_image(second_image)

    diff = np.square(reshaped_first - reshaped_second)
    pixel_difference = np.sqrt(np.sum(diff, axis=1))
    how_many_pixels_correct = np.count_nonzero( pixel_difference < threshold )
    total_pixels = pixel_difference.shape[0]
    return float(how_many_pixels_correct) / float(total_pixels)