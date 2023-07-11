# System (Default)
import sys
#   Add access if it is not in the system path.
if '../' + 'src' not in sys.path:
    sys.path.append('../..')
# Numpy (Array computing) [pip3 install numpy]
import numpy as np
# OS (Operating system interfaces)
import os
# SciencePlots (Matplotlib styles for scientific plotting) [pip3 install SciencePlots]
import scienceplots
# Matplotlib (Visualization) [pip3 install matplotlib]
import matplotlib.pyplot as plt
import matplotlib.patches as pat
# Custom Script:
#   ../Lib/Interpolation/B_Spline/Core
import Lib.Interpolation.B_Spline.Core as B_Spline

def main():
    #x = np.array([ 0. ,  1.2,  1.9,  3.2,  4. ,  6.5])
    #y = np.array([ 0. ,  2.3,  3. ,  4.3,  2.9,  3.1])

    #x = np.arange(0, 10, 0.1)
    #y  = 10 - 0.2 * x**2 + 2 * np.cos(x) + np.random.norm(0.0, 1.0, len(x))

    # Check: start in 0,0
    P = np.array([[1, 1], [2, -1], [3, -3], [4, 2], [5, 1]])
    #P = np.array([[0, 0], [0, 1]])
    n = 2

    S_Cls = B_Spline.B_Spline_Cls(n, P, 'Chord-Length', 250)
    S = S_Cls.Interpolate()

    S_new = np.zeros(S.shape)
    for i, S_i in enumerate(S):
        S_new[i, :] = S_i + np.random.uniform((-1) * np.random.uniform(0.05, 0.20), 
                                              np.random.uniform(0.05, 0.20), S.shape[1])
    S_new[0] = P[0]; S_new[-1] = P[-1]
    
    S_01 = B_Spline.B_Spline_Cls(n, S_new, 'Chord-Length', 100)
    S_Opt_Interp = S_01.Optimize_Control_Points(5)
    S_New_Opt = S_Opt_Interp.Interpolate()
    #S_dot = S_Cls.Derivative_1st()
    #print(S_dot)
    #print(S_Cls.Get_Arc_Length())
    #S_Opt_Interp = S_Cls.Optimize_Control_Points(3)
    #S_new = S_Opt_Interp.Interpolate()

    _, axis = plt.subplots()
    plt.plot(P[:, 0], P[:, 1], "--s", label="control points")
    plt.plot(S[:, 0], S[:, 1], "-", label="B-spline 1")
    plt.plot(S_new[:, 0], S_new[:, 1], "o", label="B-spline 1")
    plt.plot(S_New_Opt[:, 0], S_New_Opt[:, 1], "-", label="B-spline 1")
    plt.legend()
    plt.title("B spline")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.show()

if __name__ == "__main__":
    sys.exit(main())