import numpy as np
import matplotlib.pyplot as plt


# defining the LinearSpline function that return coefficients
def LinearSpline(x, fx):
    """

    Parameters:
        x: 1D np.array
            data point locations
        fx: 1D np.array
            dependent variable values at corresponding x
    Returns:
        coeffs (array): coefficients of the linear spline
    """
    # length of arrays ( x and fx are assumed same "length")
    length = len(x)

    # number of segments and knots(points)
    nsegs = length - 1  # this is n
    nknts = length

    # creating A square array and b column array of 2nx2n and 2nx1, i.e two equations for each segments
    A = np.zeros((nsegs * 2, 2 * nsegs))
    b = np.zeros((nsegs * 2, 1))  # y populate this field

    # ai and bi will be solved for using lin.alg

    # 0th order continuity, the equations must be equal at data points
    # there are two equations for every data point except for first and last data points
    # ie 2n-2 equations where n is the number of segments or intervals
    # need to populate A with all segment equation from 1 to 2n-1
    # yi = ai * xi + bi, and yi = a(i+1) * xi + b(i+1) for all i= i, n-1
    for i in range(1, nknts - 1):
        A[2 * i - 1, 2 * (i - 1)] = x[i]  # xi of first equation
        A[2 * i - 1, 2 * (i - 1) + 1] = 1  # 1, multiplier of bi
        A[2 * i, 2 * (i - 1) + 2] = x[i]  # xi of second equation
        A[2 * i, 2 * (i - 1) + 3] = 1  # 1, multiplier of b(i+1)

        # populating the RHS of the equation
        b[2 * i - 1] = fx[i]
        b[2 * i] = fx[i]

        # print(A)
        # print("==============================================")
        # print(b)

    # termination conditions, the first and last segment equations must pass first
    # and last data point respectively
    A[0, 0] = x[0]  # x0
    A[0, 1] = 1  # 1, multiplier of b0
    b[0] = fx[0]  # first row of RHS

    A[2 * nsegs - 1, -1] = 1  # 1, multiplier of b(n)
    A[2 * nsegs - 1, -2] = x[nknts - 1]  # xn for last termination equation
    b[2 * nsegs - 1] = fx[nknts - 1]  # last RHS element

    # print(A)
    # print("==============================================")
    # print(b)

    coeffs = np.linalg.solve(A, b)  # solve the equation

    return coeffs


# x = np.array([1,2,3,4])
# fx = np.array([2,4,3,8])
#
# print(LinearSpline(x,fx))

# creating the function for LinearSplineInterp
def LinearSplineInter(x, xnew, coeffs):
    """

    Parameters:
        x (array): data point values given, 1d row
        xnew (array): new data points to approximate with spline, 1d row
        coeffs (array): coefficients achieved from LinearSpline, 1d column

    Returns:
        (list):fapprox - dependent variable approximations at the new locations

    """

    fapprox = []  # return argument need not be a numpy.ndarray
    lenx = len(x)  # determine how many data points we have

    for i in range(0, len(xnew)):  # cycle through all location points to approximate

        for j in range(lenx):  # logic to see which equation to use in approximation

            if x[j - 1] <= xnew[i] <= x[j]:  # if the value of the location is between two data points

                # incase it equal to one the coefficients for segment should still work
                fapprox.append(
                    coeffs[j * 2 - 2][0] * xnew[i] + coeffs[j * 2 - 1][0])  # calculating the approximation and
                # appending

                # it to the list
                # if i == 0 or i == 66 or i == 99:
                #     print("a: {}, x:{}, b:{}, i: {}".format(coeffs[j * 2 - 2][0], xnew[i], coeffs[j * 2 - 1][0], i))
                break  # no need to re look again to if x new fits anywhere

            # not sure if we should allow for extrapolation so adding logic
            # that if location is outside bounds of data points
            # to append a nonsense number
            if xnew[i] < x[0] or xnew[i] > x[-1]:
                fapprox.append(np.NaN)

    return fapprox


x = np.array([3.0, 4.5, 7.0, 9.0])
fx = np.array([2.5, 1.0, 2.5, 0.5])
xnew = np.linspace(3.0, 9.0, 100)

coeffs = LinearSpline(x, fx)
fapprox = LinearSplineInter(x, xnew, coeffs)
# print(len(fapprox))
# # print(xnew)
# print(xnew[0])
# print(fapprox[0])
# print(xnew[33*2])
# print(fapprox[33*2])
# print(xnew[99])
# print(fapprox[99])

plt.plot(x, fx, 'g*', label = 'Data Points')
plt.plot(xnew,fapprox,'r', label='Linear interpolated approximations')

plt.legend()
# plt.savefig("A8.png")
plt.show()

