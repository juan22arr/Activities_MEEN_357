import math

import matplotlib.pyplot as plt

import numpy as np

# true solution
# T = 200 * (1 - log_0.5(r))

a = 1  # default state
# 10 nodes so n = 9
n = 9

# equation t'' = -1/r * t'
# y'' = (y_(i+1) - 2y_i +y_(i-1))/h^2 (1)
# y' = (y_(i+1)-y_(i-1))/ 2h (2)
# y_0 = 0, y_10 = 200 degC
# x_0 = a/2, x_10 = a
# assuming evenly spaced nodes h = (a- .5a)/9
h = a / 18
# using equation 1 and 2 (y is T, x is r)
# 0 = (2hr+h^2)y_(i+1) - 4hr*y_i + (2hr-h^2)y_(i-1)

# creating A matrix
A = np.zeros((n + 1, n + 1))  # a 10 by 10 matrix so 10 nodes
A[0, 0] = 1  # y_0 = 0
A[n, n] = 1  # y_9 = 200
x = np.linspace(0.5, 1,
                10)  # setting an x array to get values from for all x values from .5 to 1, of 10 values including endpoint

# print(x)

# setting the square array with coefficients of equation  (2hr+h^2)y_(i+1) - 4hr*y_i + (2hr-h^2)y_(i-1)
for i in range(1, n):
    A[i, i - 1] = 2 * h * x[i] + h ** 2
    A[i, i] = - 4 * h * x[i]
    A[i, i + 1] = 2 * h * x[i] - h ** 2

# print(A)

# creating the 'solution' matrix where all values are zero per the equation and y_0 = 0 except y_9 = 200
b = np.zeros((n + 1))
b[-1] = 200  # setting y_9 = 200
# print(b)

y = (np.linalg.solve(A, b))  # y values
# print(y)

y_true = 200 * (1 - np.log(x) / np.log(0.5))

plt.plot(x, y, 'g', label='Finite Difference method')
plt.plot(x, y_true, 'b', label='Analytical solution')
plt.xlabel("Radius")
plt.ylabel("Temperature [C]")
plt.legend()
plt.show()
# plt.savefig("task3.png")
