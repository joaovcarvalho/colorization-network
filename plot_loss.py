import re
import matplotlib.pyplot as plt
import numpy as np

file = open('loss.out', 'r')
lines = [
    re.sub("[^0-9]", "", line) for line in file.readlines()
]

numbers = map(float, lines)
data = np.array(numbers)

plt.plot(data)
plt.ylabel('loss')
plt.show()
