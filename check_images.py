import multiprocessing
import os
from functools import partial

import numpy as np
from PIL import Image

from image_preprocessing import _count_valid_files_in_directory, _list_valid_filenames_in_directory

directory = 'imagenet'

classes = []

white_list_formats = {'png', 'jpg', 'jpeg', 'bmp', 'ppm', 'tif', 'tiff'}

for subdir in sorted(os.listdir(directory)):
    if os.path.isdir(os.path.join(directory, subdir)):
        classes.append(subdir)
num_classes = len(classes)
class_indices = dict(zip(classes, range(len(classes))))

pool = multiprocessing.pool.ThreadPool()
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


def check_image(filename, directory):
    final_path = os.path.join(directory, filename)
    try:
        Image.open(final_path)
    except IOError:
        try:
            print('Removing file -> {}'.format(filename))
            os.remove(final_path)
        except OSError:
            pass


pool.map(partial(check_image, directory=directory), filenames)