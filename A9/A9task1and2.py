import math

import numpy as np

def gs_max(fhandle, xlb, xub, Tolx = .000001):
    """
    This function uses the golden section algorithm to find the maximum of a function of one variable. The
    function halts when the approximated location of the maximum does not change much between
    iterations (the difference falls below a threshold).

    calling syntax:
    xstar, fval = gs_max(fhandle, xlb, xub)
    xstar, fval = gs_max(fhandle, xlb, xub, TolX)

    :param fhandle: function handle
                    Handle to objective function to be maximized, f(x).
    :param xlb: scalar
                Lower bound of search domain
    :param xub: scalar
                Upper bound of search domain
    :param Tolx: positive scalar
                 Convergence tolerance. Function terminates when (1-R)|(xub-xlb)/xstar| <
                 Tolx, where R is the golden ratio. Default 1e-6
    :return xstar: scalar
                    Location of maximum
    :return fval: scalar
                    Value of objective function at xstar
    """

    # validating that first input is a handle to a function
    # input two and three are numeric scalars
    # and if a Tolx was passed that it is a positive scalar
    # if not then raise an exception
    if not callable(fhandle):
        raise Exception("First input is not a function handle")

    if not (isinstance(xlb, float) or isinstance(xlb, int)):
        raise Exception("Second input is not a scalar")
    if not (isinstance(xub, float) or isinstance(xub, int)):
        raise Exception("Third input is not a scalar")

    if not(isinstance(Tolx,int) or isinstance(Tolx, float)) or Tolx <= 0:
        raise Exception("The fourth input must be a positive scalar: {}".format(Tolx))

    # defining the golden ratio
    R = (np.sqrt(5)-1) / 2
    difference = (1-R)*abs(xub - xlb) #for the first loop we will use this value since we have no xstar
    xstar = 9999999
    fstar = 9999999 # nonsense numbers meaning something wasn't found
    # print(difference)

    while (difference - Tolx) >= 0:
        # defining the step to check bounds with
        d = R * (xub - xlb)

        # defining what the new bounds could be
        x1 = xlb + d
        x2 = xub - d

        # the update rule, it is unlikely the functions will ever be equal
        # so else can be used to denote f(x2) > f(x1)
        # and if they are equal just lower the upper bound
        # update xstar with greater value of the two
        if fhandle(x1) > fhandle(x2):
            xlb = x2
            xstar = x1
            fstar = fhandle(x1)
        else:
            xub = x1
            xstar = x2
            fstar = fhandle(x2)

        # don't know what  to do if gets stuck on xstar being 0
        if xstar == 0:
            break
        difference = (1-R)*abs((xub - xlb) / xstar)
        # print(difference)

    return xstar, fstar

#
# f = lambda x: -(x-2)**2
#
# print(gs_max(f,0,4))


def gradient_ascent_2D(fhandle, x0, Tolx = .000001, MaxIter = 1000):
    """

    :param fhandle: function handle
                Handle to objective function to be maximized, f(x).
    :param x0: 1x2 vector
                Initial guess
    :param Tolx: scalar > 0 (optional)
                Convergence tolerance. Function terminates when norm of change in search
                vector is less than this amount. Default is 1e-6.
    :param MaxIter: scalar > 0
                (optional) Maximum number of iterations. Default is 1000.
    :return xstar: scalar
                Location of maximum
    :return fval: scalar
                Value of objective function at its maximum.
    :return exitFlag: scalar
                1 if successful convergence to maximum, 0 if return due to iteration
                timeout.
    :return iter_num: scalar Number of iterations until convergence
    """

    # default value of returns meaning nothing was executed
    xstar = x0
    fstar = 9999
    exitFlag = -1
    iter_num = -1

    # validating inputs
    if not callable(fhandle):
        raise Exception("First input is not a function handle")

    try:
        if x0.shape != (2,): # a 2 element numpy array will return that argument when asked for shape
            raise Exception("The second input must be two element (1x2) vector (numpy array) ")
    except:
        raise Exception("The second input must be two element (1x2) vector (numpy array) ")

    if not (isinstance(Tolx, int) or isinstance(Tolx, float)) or Tolx <= 0:
        raise Exception("The fourth input must be a positive scalar")

    if not (isinstance(MaxIter, int)) or MaxIter <= 0:
        raise Exception("The fifth input must be a positive integer")

    # define finite differecne step size for x1, and x2(y)
    if x0[0] <= .000001 or x0[1] <= .000001:
        h1 = .000001
        h2 = .000001
    else:
        h1 = .000001 * x0[0]
        h2 = .000001 * x0[1]

    # creating short hand

    x = xstar[0]
    y = xstar[1]

    # creating short hand for the finite step sizes
    x_forward = np.array([x + h1 * x, y])
    x_backward = np.array([x - h1 * x, y])
    y_forward = np.array([x, y + h2 * y])
    y_backward = np.array([x, y - h2 * y])

    # calculating the first partial derivatives using finite differences
    dfx = (fhandle(x_forward) - fhandle(x_backward)) / (2 * h1 * x)
    dfy = (fhandle(y_forward) - fhandle(y_backward)) / (2 * h2 * y)

    # creating a new func to be called by gs_max
    new_func = lambda h: fhandle(np.array([x + dfx * h,y + dfy * h]))

    # calling gs_max to find optimal step size
    opt_step, _ = gs_max(new_func, 0, 1)

    # updating new location and maximum
    new_x = x + opt_step * dfx
    new_y = y + opt_step * dfy
    old_x0 = xstar
    xstar = np.array([new_x, new_y])
    fstar = fhandle(xstar)

    # checking if tolerance reached
    numerator = np.linalg.norm(xstar - old_x0)
    denomerator =  np.linalg.norm(old_x0)
    tol_check = numerator/denomerator
    # print(tol_check)

    iter_num = 1

    while tol_check >= Tolx:
        if iter_num >= MaxIter:
            exitFlag = 0
            return xstar, fstar, exitFlag, iter_num-1
        x = xstar[0]
        y = xstar[1]

        # creating short hand for the finite step sizes
        x_forward = np.array([x + h1 * x, y])
        x_backward = np.array([x - h1 * x, y])
        y_forward = np.array([x, y + h2 * y])
        y_backward = np.array([x, y - h2 * y])

        # calculating the first partial derivatives using finite differences
        dfx = (fhandle(x_forward) - fhandle(x_backward)) / (2 * h1 * x)
        dfy = (fhandle(y_forward) - fhandle(y_backward)) / (2 * h2 * y)

        # creating a new func to be called by gs_max
        new_func = lambda h: fhandle(np.array([x + dfx * h, y + dfy * h]))

        # calling gs_max to find optimal step size
        opt_step, _ = gs_max(new_func, 0, 1)

        # updating new location and maximum
        new_x = x + opt_step * dfx
        new_y = y + opt_step * dfy
        old_x0 = xstar
        xstar = np.array([new_x, new_y])
        fstar = fhandle(xstar)

        # checking if tolerance reached
        numerator = np.linalg.norm(xstar - old_x0)
        denomerator = np.linalg.norm(old_x0)
        tol_check = numerator / denomerator

        # print(tol_check)

        iter_num += 1

    exitFlag = 1
    return xstar, fstar, exitFlag, iter_num

# gradient_ascent_2D(print, np.array([1,2]))

# print(np.array([1,2]).shape != (2,1))
