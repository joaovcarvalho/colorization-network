import sys
import matplotlib.pyplot as plt

import numpy as np

def plot_graph(data):
    plt.subplot(2, 1, 2)
    x = range(len(data))
    plt.ylim(0, 100)
    plt.plot(x, data)
    plt.xlabel('Limite(t)')
    plt.ylabel('Acuracia(%)')

final_accuracies = np.load('accuracies.npy')
averages = np.mean(final_accuracies, axis=0) * 100
print(averages)
plot_graph(averages)
plt.show()

