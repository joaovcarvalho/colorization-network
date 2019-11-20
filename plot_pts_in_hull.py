import os

import matplotlib.gridspec as gridspec
import matplotlib.pylab as plt
import numpy as np

from quantization import convert_quantization_to_image

data_dir = 'resources/colorful'

# Load the gamut points location
# q_ab = np.load(os.path.join(data_dir, "pts_in_hull.npy"))
# print(q_ab)

BINS = 16
MAX_BINS = 256

examples = np.zeros((BINS, BINS, MAX_BINS))

for i in range(BINS):
    for j in range(BINS):
        examples[i][j][i * BINS + j] = 1.0

q_ab = convert_quantization_to_image(examples, BINS)
q_ab = q_ab.reshape((MAX_BINS, 2))
q_ab /= 255
q_ab *= 200
q_ab -= 100

plt.figure(figsize=(8, 8))
gs = gridspec.GridSpec(1, 1)
ax = plt.subplot(gs[0])
for i in range(q_ab.shape[0]):
    ax.scatter(q_ab[:, 0], q_ab[:, 1])
    ax.annotate(str(i), (q_ab[i, 0], q_ab[i, 1]), fontsize=6)
    ax.set_xlim([-110, 110])
    ax.set_ylim([-110, 110])

plt.show()
