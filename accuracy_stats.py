import numpy as np

number = True
numbers = []
while number != None:
    try:
        number = input()
        numbers.append(number)
        print(number)
    except EOFError:
        number = None

arr = np.array(numbers)
print('Average', np.average(arr))
print('Std', np.std(arr))
print('Median', np.median(arr))