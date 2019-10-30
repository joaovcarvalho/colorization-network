import os

import matplotlib.gridspec as gridspec
import matplotlib.pylab as plt
import numpy as np

data_dir = 'resources/colorful'

# Load the gamut points location
q_ab = np.load(os.path.join(data_dir, "pts_in_hull.npy"))

plt.figure(figsize=(15, 15))
gs = gridspec.GridSpec(1, 1)
ax = plt.subplot(gs[0])
for i in range(q_ab.shape[0]):
    ax.scatter(q_ab[:, 0], q_ab[:, 1])
    ax.annotate(str(i), (q_ab[i, 0], q_ab[i, 1]), fontsize=6)
    ax.set_xlim([-110, 110])
    ax.set_ylim([-110, 110])

plt.show()
