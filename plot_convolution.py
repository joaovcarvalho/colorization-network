import numpy as np
import matplotlib.pylab as plt

x = np.linspace(0, 10, 400)
data = np.piecewise(x, [x < 4, x > 4, x > 6], [0, 1, 0])

limit = 0

convolution_data = []
count = 0
for i in x:
    kernel = np.piecewise(x, [x < limit - 1 + i, x > limit - 1 + i, x > limit + 1 + i], [0, 1, 0])
    convolution = np.convolve(data, kernel, mode="valid") / 80
    count += 1
    convolution_data.append(convolution)
    convolution_space = np.linspace(0, i, count)

    plt.plot(x, data)
    plt.plot(x, kernel)
    plt.plot(convolution_space, np.array(convolution_data))
    plt.xlabel('x')
    plt.ylabel('y')
    plt.axis('tight')
    plt.savefig('convolution/frame_{}.png'.format(count))
    plt.clf()
