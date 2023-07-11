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

    n = 30
    x = np.sort(np.random.uniform(0, 5, size=n))
    y = np.sin(x) + 0.1*np.random.randn(n)

    # Check: start in 0,0
    #P = np.array([x, y]).T
    P = np.array([[0, 0], [0, 1]])
    n = 1

    S_Cls = B_Spline.B_Spline_Cls(n, P, 'Chord-Length', 100)
    S = S_Cls.Interpolate()

    #S_dot = S_Cls.Derivative_1st()
    #print(S_dot)
    print(S_Cls.Get_Arc_Length())
    #S_Opt_Interp = S_Cls.Optimize_Control_Points(3)
    #S_new = S_Opt_Interp.Interpolate()

    _, axis = plt.subplots()
    plt.plot(P[:, 0], P[:, 1], "--s", label="control points")
    #plt.plot(S_Opt.P[:, 0], S_Opt.P[:, 1], "--o", label="control points opt.")
    plt.plot(S[:, 0], S[:, 1], "-", label="B-spline 1")
    #plt.plot(S_new[:, 0], S_new[:, 1], "-", label="B-spline 1")
    plt.legend()
    plt.title("B spline")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.show()

if __name__ == "__main__":
    sys.exit(main())