import time
import numpy as np
import cv2


def save_aggregate_images(images_collected):
    timestr = time.strftime("%Y%m%d_%H%M%S")

    final_test_image = None
    rows_images = []

    current_image = None
    for i, image in enumerate(images_collected):
        if current_image is not None:
            current_image = np.append(current_image, image, axis=1)
        else:
            current_image = image

        if i % 3 == 2:
            rows_images.append(current_image)
            current_image = None

    for row_image in rows_images:
        print(row_image.shape)
        if final_test_image is not None:
            print(final_test_image.shape)
            final_test_image = np.append(final_test_image, row_image, axis=0)
        else:
            final_test_image = row_image

    # cv2.imshow('test', final_test_image)
    # cv2.waitKey(0)
    filename = 'results/results_{}.png'.format(timestr)
    cv2.imwrite(filename, final_test_image)

    print('Saved in: {}'.format(filename))