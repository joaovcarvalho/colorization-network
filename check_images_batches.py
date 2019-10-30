import numpy as np
from keras.preprocessing.image import ImageDataGenerator

from image_preprocessing import ColorizationDirectoryIterator
# from plot_weights import plot_weights

# data_folder = 'imagenet'
data_folder = 'places/data/vision/torralba/deeplearning/images256'

TARGET_SIZE = (64, 64)
BATCH_SIZE = 1000

train_datagen = ImageDataGenerator(
    rescale=1. / 255,
)

data_generator = ColorizationDirectoryIterator(
    data_folder,
    train_datagen,
    target_size=TARGET_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='original',
    # save_to_dir='distributions',
    # save_prefix='original'
)

NUM_BINS = 16
final_sum = np.zeros((NUM_BINS ** 2))
pixels_count = 0
batch_count = 0

SAVE_EVERY = 10

for x, quantum in data_generator:
    current_sum = np.sum(quantum, axis=(0, 1, 2))
    print('Batch {} = {}'.format(batch_count, current_sum.shape))
    assert current_sum.shape == (NUM_BINS ** 2,)

    # file_name = 'distributions/distributions_{}'.format(batch_count)
    # plot_weights(current_sum, save_fig_to=file_name)

    final_sum = final_sum + current_sum
    quantum_pixels = quantum.shape[0] * quantum.shape[1] * quantum.shape[2]
    pixels_count = pixels_count + quantum_pixels
    batch_count += 1

    if batch_count % SAVE_EVERY == 0:
        assert (np.sum(final_sum) == pixels_count)
        print('Saving weights...')
        weights = final_sum / pixels_count
        np.save('weights.npy', weights)

assert(np.sum(final_sum) == pixels_count)
weights = final_sum / pixels_count
np.save('weights.npy', weights)

print('Finished...')
