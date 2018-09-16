import numpy as np
import matplotlib.pyplot as plt

weights = np.load('weights.npy')
x = np.arange(weights.shape[0])

fig, ax = plt.subplots()
plt.bar(x, weights)
plt.show()
