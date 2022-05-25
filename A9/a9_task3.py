from A9task1and2 import *
import numpy as np

def func1(x0):
    return -(x0[0] + 1)**2 - (x0[1] + 3)**2

def func2(x0):
    return -100*(x0[0] + 1)**2 - (x0[1] + 3)**2

print(gradient_ascent_2D(func1,np.array([-40,-10])))

print(gradient_ascent_2D(func2, np.array([-40,-10])))