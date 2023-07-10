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
import Lib.Interpolation.Bezier.Core as Bezier

def main():
    P = np.array([[0, 0], [1, 1], [2, -1], [3, 0], [4, 2], [5, 1]])
    n = 2

    S_Cls = B_Spline.B_Spline_Cls(n, P, 'Chord-Length', 1000)
    S = S_Cls.Interpolate()

    Bezier_0 = Bezier.Bezier_Cls('Explicit', S, 100)
    B_t_0 = Bezier_0.Interpolate()

    #S_dot = S_Cls.Derivative_1st()
    #print(S_dot)
    #print(S_Cls.Get_Arc_Length())
    #S_Opt_Interp = S_Opt.Interpolate()

    _, axis = plt.subplots()
    plt.plot(P[:, 0], P[:, 1], "--s", label="control points")
    #plt.plot(S_Opt.P[:, 0], S_Opt.P[:, 1], "--o", label="control points opt.")
    plt.plot(S[:, 0], S[:, 1], "-", label="B-spline 1")
    plt.plot(B_t_0[:, 0], B_t_0[:, 1], "-", label="B-spline 2")
    plt.legend()
    plt.title("B spline")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.show()

if __name__ == "__main__":
    sys.exit(main())