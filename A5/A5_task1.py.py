# analytical solution
# two eulers method
# one for midpoint
# One fir RK4

import sympy as spy
import numpy as np
import matplotlib.pyplot as plt


# analytical solution solved by differential methods
def analytical_solution(t):
    return np.exp(0.25 * t ** 4 - 1.5 * t)


time = np.linspace(0, 2)
plt.plot(time, analytical_solution(time), label='Analytical solution')


# plt.show()


# creating euler method function, y_i+1 = y_i + fun(xi,yi)*h, fun is the first derivation of y
# Define a function to solve the IVP using Euler's method

def euler_ivp_solver(fun, x0, y0, xStop, h):
    """
    Inputs
    ----------
    fun : callable
        ODE function (i.e., dydx function)
    x0 : scalar
        Initial value of x
    y0 : scalar
        Initial value of y (at x0)
    xStop : scalar
        value of x where we want to stop integrating
    h : scalar
        step size

    Outputs
    -------
    X : list
        list of x values
    Y : list
        list of y values corresponding to entries of X

    """
    x_i = x0 + 0  # to ensure it diffent var
    y_i = y0 + 0  # ^
    X = []
    Y = []

    while x_i <= xStop:
        X.append(x_i)
        Y.append(y_i)

        # y_i is evaluated first because x_i will be updated to x_i+1 and we don't use
        # the updated x_i in the approximation of y_i+1 we use x_i
        y_i = y_i + fun(x_i, y_i) * h
        x_i = x_i + h

    return X, Y


# defining the function to be used for Euler's method
def my_fun(t, y):
    return y * t ** 3 - 1.5 * y


# Solve our test function and plot

h = 0.5  # step size
X, Y = euler_ivp_solver(my_fun, 0, 1, 2, h)

plt.plot(X, Y, color='maroon', label="Euler's method h=0.5")

h = 0.25  # step size
X, Y = euler_ivp_solver(my_fun, 0, 1, 2, h)

plt.plot(X, Y, color='blue', label="Euler's method h=0.25")


# plt.show()

def mid_point(fun, x0, y0, xStop, h):
    """
    Inputs
    ----------
    fun : callable
        ODE function (i.e., dydx function)
    x0 : scalar
        Initial value of x
    y0 : scalar
        Initial value of y (at x0)
    xStop : scalar
        value of x where we want to stop integrating
    h : scalar
        step size

    Outputs
    -------
    X : list
        list of x values
    Y : list
        list of y values corresponding to entries of X

    """

    x_i = x0 + 0  # to ensure it diffent var
    y_i = y0 + 0  # ^
    X = []
    Y = []

    while x_i <= xStop:
        X.append(x_i)
        Y.append(y_i)

        # y_i is evaluated first because x_i will be updated to x_i+1 and we don't use
        # the updated x_i in the approximation of y_i+1 we use x_i
        y_i = y_i + fun(x_i + h / 2, y_i + h / 2 * fun(x_i, y_i)) * h  # this is the explicit function of the
        # midpoint method
        x_i = x_i + h

    return X, Y


h = 0.5  # try out different values to see how the solution changes
X, Y = mid_point(my_fun, 0, 1, 2, h)

plt.plot(X, Y, color='red', label="Midpoint method h=0.5")


# plt.show()


def RK_4(fun, x0, y0, xStop, h):
    """
    Inputs
    ----------
    fun : callable
        ODE function (i.e. dy/dx function)
    x0 : scalar
        Initial value of x
    y0 : scalar
        Initial value of y (at x0)
    xStop : scalar
        value of x where we want to stop integrating
    h : scalar
        step size

    Outputs
    -------
    X : list
        list of x values
    Y : list
        list of y values corresponding to entries of X

    """

    x_i = x0 + 0  # to ensure it diffent var
    y_i = y0 + 0  # ^
    X = []
    Y = []

    while x_i <= xStop:
        X.append(x_i)
        Y.append(y_i)

        # defining the k values of RK - 4
        k_1 = fun(x_i, y_i)
        k_2 = fun(x_i + .5 * h, y_i + .5 * k_1 * h)
        k_3 = fun(x_i + .5 * h, y_i + .5 * k_2 * h)
        k_4 = fun(x_i + h, y_i + k_3 * h)

        # y_i is evaluated first because x_i will be updated to x_i+1 and we don't use
        # the updated x_i in the approximation of y_i+1 we use x_i
        y_i = y_i + h / 6 * (k_1 + 2 * k_2 + 2 * k_3 + k_4)  # this is the explicit function of the
        # RK-4 method
        x_i = x_i + h

    return X, Y


h = 0.5  # step size
X, Y = RK_4(my_fun, 0, 1, 2, h)  # x_0 = 0, y_0 = 1, 0 to 2

plt.plot(X, Y, color='green', label="RK-4, h = 0.5")
plt.legend()
# plt.savefig("solutins.svg")

plt.show()
