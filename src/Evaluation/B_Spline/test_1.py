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
# Custom Script:
#   ../Lib/Interpolation/B_Spline/Core
import Lib.Interpolation.B_Spline.Core as B_Spline

def main():
    P = np.array([[0, 0], [1, 1], [2, -1], [3, 0], [4, 2], [5, 1]])
    n = 1

    S_Cls = B_Spline.B_Spline_Cls(n, P, 'Chord-Length', 100)
    S = S_Cls.Interpolate()

    #S_dot = S_Cls.Derivative_1st()
    #print(S_dot)
    #print(S_Cls.Get_Arc_Length())

    fig = plt.figure("B-spline curve", figsize = (6, 3))
    plt.plot(P[:, 0], P[:, 1], "--s", label="control points")
    plt.plot(S[:, 0], S[:, 1], "-", label="B-spline 1")
    plt.legend()
    plt.title("B spline")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.show()

if __name__ == "__main__":
    sys.exit(main())