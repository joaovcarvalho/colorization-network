from multiprocessing.pool import ThreadPool

import cv2
import multiprocessing
import os
import traceback
from functools import partial

import numpy as np
from PIL import Image

from image_preprocessing import \
    _count_valid_files_in_directory, \
    _list_valid_filenames_in_directory
from quantization import quantize_lab_image

# directory = 'imagenet'
# directory = 'places/data/vision/torralba/deeplearning/images256'
directory = '../places-dataset'

classes = []

white_list_formats = {'png', 'jpg', 'jpeg', 'bmp', 'ppm', 'tif', 'tiff'}

for subdir in sorted(os.listdir(directory)):
    if os.path.isdir(os.path.join(directory, subdir)):
        classes.append(subdir)

num_classes = len(classes)
class_indices = dict(zip(classes, range(len(classes))))

print('classes: {}'.format(classes))

pool = ThreadPool()
function_partial = partial(_count_valid_files_in_directory,
                           white_list_formats=white_list_formats,
                           follow_links=False,
                           split=None)

samples = sum(pool.map(function_partial,
                       (os.path.join(directory, subdir)
                        for subdir in classes)))

# second, build an index of the images in the different class subfolders
results = []

filenames = []
final_classes = np.zeros((samples,), dtype='int32')
i = 0
for dirpath in (os.path.join(directory, subdir) for subdir in classes):
    results.append(pool.apply_async(_list_valid_filenames_in_directory,
                                    (dirpath, white_list_formats, None,
                                     class_indices, False)))
for res in results:
    classes, filenames = res.get()
    final_classes[i:i + len(classes)] = classes
    filenames += filenames
    i += len(classes)

pool.close()
pool.join()

pool = ThreadPool()

lock = multiprocessing.Lock()

NUM_BINS = 16

final_sum = 0.0
pixels_count = 0
image_count = 0

PRINT_EVERY_N_IMAGES = 100


def check_image(filename, directory):
    global final_sum, pixels_count, image_count
    print(filename)

    final_path = os.path.join(directory, filename)
    try:
        image = Image.open(final_path)
        rgb_im = image.convert('RGB')
        x = np.asarray(rgb_im)

        x = x.astype('uint8')
        x = cv2.cvtColor(x, cv2.COLOR_RGB2LAB)
        x = x.astype('float')

        subject = np.abs(x[:, :, 0] - 119.85)
        assert subject.shape == (256, 256)

        # subject = quantize_lab_image(x, NUM_BINS, 256)
        sum = np.sum(subject)

        with lock:
            final_sum += sum
            pixels_count += subject.shape[0] * subject.shape[1]
            image_count += 1

            if image_count % PRINT_EVERY_N_IMAGES == 0:
                weights = np.sqrt(final_sum / pixels_count)
                print(weights)
                np.save('std.npy', weights)

    except Exception as e:
        print(e)
        traceback.print_exc()


print('How many images: {}'.format(len(filenames)))
pool.map(partial(check_image, directory=directory), filenames)
pool.join()

# weights = final_sum / pixels_count
# np.save('weights', weights)
