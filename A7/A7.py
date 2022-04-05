
import numpy as np
import matplotlib.pyplot as plt
import copy



def GaussElim_Solver(A, b):
    # assuming A is nxn, and b is nx1 rep as 1xn

    num_rows, num_cols = A.shape  # getting number of rows and columns
    # making a and b hold floats
    A_new = A.astype(np.float32)
    # print(A_new)
    b_new = b.astype(np.float32)

    # gaussian elimination forward step
    for i in range(1,num_rows):
        for j in range(i, num_rows):
            coe = A_new[j][i-1]/A_new[i-1][i-1]  # finding coefficient to multiply the row containing the next x we're trying to isolate
            # subtracting row - row_base * coe
            A_new[j] = A_new[j] - np.multiply(coe, A_new[i-1])
            b_new[j][0] = b_new[j][0] - np.multiply(coe, b_new[i-1])

    x = np.zeros((num_cols, 1))  # nx1

    # back solving through the rows
    x[num_cols - 1] = b_new[num_cols-1][0]/A_new[num_cols-1][num_cols-1]  # final xn
    for i in range(num_cols - 2, -1, -1):
        sum = b_new[i][0]
        for j in range(i+1,num_cols):
            sum = sum - np.multiply(A_new[i][j], x[j])  # creating the left part of x(n) * a(n) = b - ...
        x[i] = sum / A_new[i][i]
    return x



# cramer's rule
def Cramer_solver(A, b):
    # assuming A is nxn and b is nx1
    # and both are np,arrays
    # making A and b into float arrays
    num_rows, num_cols = A.shape

    A_f = A.astype(np.float32)
    b_f = b.astype(np.float32)

    # finding determinant of A assuming exists and is square
    D = np.linalg.det(A_f)
    if D == 0 or D == 0.0:
        return np.NaN  # no solution
    else:
        x_det = [] # list to hold x's
        for i in range(num_cols): # loop through all cols
            A_mod = copy.deepcopy(A_f)  # make a temp deep copy as to not modify original
            for j in range(num_cols): # loop through all rows (should be equal to number of cols)
                A_mod[j][i] = b_f[j][0]  # modify the matrix with b
            x_det.append(np.linalg.det(A_mod) / D)  # solve for det of modded A divide by the det of A and append vlaue to
            # list x

    return x_det

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
x = np.linspace(0.5, 1, 10)  # setting an x array to get values from for all x values from .5 to 1, of 10 values including
# endpoint

# print(x)

# setting the square array with coefficients of equation  (2hr+h^2)y_(i+1) - 4hr*y_i + (2hr-h^2)y_(i-1)
for i in range(1, n):
    A[i, i - 1] = 2 * h * x[i] - h ** 2
    A[i, i] = - 4 * h * x[i]
    A[i, i + 1] = 2 * h * x[i] + h ** 2

# print(A)

# creating the 'solution' matrix where all values are zero per the equation and y_0 = 0 except y_9 = 200
b = np.zeros((n + 1,1))
b[-1] = 200  # setting y_9 = 200
# print(b)

y = (GaussElim_Solver(A, b))  # y values using gauss elim
y_cr = (Cramer_solver(A,b))   # solving using cramers rule
# print(y)
y_1 = np.linalg.solve(A,b)   # solved using numpy
y_true = 200 * (1 - np.log(x) / np.log(0.5))   # analytical solution

plt.plot(x, y, 'g', label='Gaussian Elimination')
plt.plot(x, y_true, 'b', label='Analytical solution')
plt.plot(x, y_1, 'c', label='Finite difference method')
plt.plot(x, y_cr, 'r', label='Cramer\'s rule')
plt.xlabel("Radius")
plt.ylabel("Temperature [C]")
plt.legend()
plt.show()

# print(y_cr[1])

# # creating table with values somewhat formatted, to be silenced in submission
# for i in range(11):
#     if (i==0):
#         print("x | y analytical | y lin.solve | y Gaussian | y Cramer's Rule")
#     else:
#         k = i -1
#         x_f = float(x[k])
#         y_true_f = float(y_true[k])
#         y_1_f = float(y_1[k])
#         y_f = float(y[k][0])
#         y_cr_f = float(y_cr[k])
#         print('{0:1.5f} | {1:1.5f} | {2:1.5f} | {3:1.5f} | {4:1.5f}'.format(x_f,y_true_f, y_1_f, y_f, y_cr_f))
