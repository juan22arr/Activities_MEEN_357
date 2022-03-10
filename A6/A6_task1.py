import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp


# adding rk4 function for A5
def RK_4_2nd_ODE(fun, y, x_start, xStop, h):
    """

    ----------
    fun : callable
        ODE function (i.e. dy/dx function)
    y : initial conditions

    x_start : scalar
        value to intialize x at
    xStop : scalar
        value of x where we want to stop integrating
    h : scalar
        step size

    Outputs
    -------
    Y1 : list
        list of y1 values
    Y2: list
        list of y2 values corresponding to entries of Y1
    X : list
        list of steps taken

    """

    Y1 = []
    Y2 = []
    time = []  # array containing the time or step size associated with values of the variables

    # getting initial variables
    t_i = x_start + 0  # ensure s different variable
    y1_i = y[0] + 0
    y2_i = y[1] + 0

    while t_i <= xStop:
        # this function is specific to this situation
        time.append(t_i)
        Y1.append(y1_i)
        Y2.append(y2_i)

        y_pass = np.array([y1_i, y2_i])

        # defining the k values of RK - 4
        k_1 = fun(t_i, y_pass)
        k_2 = fun(t_i + .5 * h, y_pass + .5 * k_1 * h)
        k_3 = fun(t_i + .5 * h, y_pass + .5 * k_2 * h)
        k_4 = fun(t_i + h, y_pass + k_3 * h)

        # y_i is evaluated first because t_i will be updated to t_i+1 and we don't use
        # the updated t_i in the approximation of y_i+1 we use t_i
        y_pass = y_pass + h / 6 * (k_1 + 2 * k_2 + 2 * k_3 + k_4)  # this is the explicit function of the
        # RK-4 method
        t_i = t_i + h

        y1_i = y_pass[0]
        y2_i = y_pass[1]

    return time, Y1, Y2


# function to describe following matrix ODE
# eq: y'' + 0.5 y' + 7y = 0
# y_1 = y
# y_2 = y'
# dy_1/dt = y_2
# dy_2/ dt = -0.5*y_2 - 7*y_1
def prob_254(t, y):
    """
    purpose of this function is to compute the derivatives of the two state variables

    Parameters
    ----------
    t : scalar
        time of state vector
    y : 2 element array
        value of state vector y[0] is first ver ans so on


    Outputs
    -------
    dydt : np.array
        Vector containing the derivatives of the two state variables
        dydt[0] is derivative of y_1, dydt[1] is derivative of y_2
    """

    dydt = np.array([y[1], -0.5 * y[1] - 7 * y[0]])  # giving the derivatives of the variables in question given by the
    # equation above

    return dydt


# defining the span
xStart = 0
xStop = 5

# defining the initial conditions of y_1 and y_2
y = np.array([4, 0])

# defining step size
h = 0.5

T, y1, y2 = RK_4_2nd_ODE(prob_254, y, xStart, xStop, h)
plt.plot(T, y1,'b')


sol = solve_ivp(prob_254, (0, 5), y, method='RK45', max_step=0.5)
t = sol.t
y0 = sol.y[0, :]

plt.plot(t, y0)
plt.show()

# defining the true value with h = 0.5^20
h = 0.5 ^ 20
sol = RK_4_2nd_ODE(prob_254, y, xStart, xStop, h)
# getting values, choosing element 2/h will give the value at 2
t = sol.t[2/h]
y0_x2 = sol.y[0, 2/h]
y1_x2 = sol.y[0, 2/h]



for i in range(1,8): # picks numbers 1 through 7
    h = 0.5 ^ i  # step size being halved exponentially
    sol = RK_4_2nd_ODE(prob_254, y, xStart, xStop, h)  # solve
