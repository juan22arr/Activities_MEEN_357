import numpy as np
import matplotlib.pyplot as plt


# defining the LinearSpline function that return coefficients
def LinearSpline(x, fx):
    """

    :param x: 1D np.array
        data point locations
    :param fx: 1D np.array
        dependent variable values at corresponding x
    :return coeffs: array
        coefficients of the linear spline
    """
    # length of arrays ( x and fx are assumed same "length")
    length = len(x)

    # number of segments and knots(points)
    nsegs = length - 1  # this is n
    nknts = length

    # creating A square array and b column array of 2nx2n and 2nx1, i.e two equations for each segments
    A = np.zeros((nsegs * 2, 2 * nsegs))
    b = np.zeros((nsegs * 2))  # y populate this field

    # ai and bi will be solved for using lin.alg

    # 0th order continuity, the equations must be equal at data points
    # there are two equations for every data point except for first and last data points
    # ie 2n-2 equations where n is the number of segments or intervals
    # need to populate A with all segment equation from 1 to 2n-1
    # yi = ai * xi + bi, and yi = a(i+1) * xi + b(i+1) for all i= i, n-1
    for i in range(1, nsegs - 1):
        A[2 * i - 1, 2 * (i - 1)] = x[i]  # xi of first equation
        A[2 * i - 1, 2 * (i - 1) + 1] = 1  # 1, multiplier of bi
        A[2 * i, 2 * (i - 1) + 2] = x[i]  # xi of second equation
        A[2 * i, 2 * (i - 1) + 3] = 1  # 1, multiplier of b(i+1)

        #

    # termination conditions, the first and last segment equations must pass first
    # and last data point respectively

    return coeffs
