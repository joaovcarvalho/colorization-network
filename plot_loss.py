import re
import matplotlib.pyplot as plt
import numpy as np

file = open('imagenet_train.out', 'r')
loss_lines = [line for line in file.readlines() if "loss" in line]
loss_line = ''.join(loss_lines)
lines = [line.split()[-1] for line in loss_line.split("\x08") if line is not '']

lines = [re.sub("[^0-9\.]", "", line) for line in lines]
lines = [line for line in lines if line != '']

numbers = map(float, lines)
data = np.array(numbers)
data = data / np.max(data)

plt.plot(data)
plt.ylabel('loss')
plt.show()

