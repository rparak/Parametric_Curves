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
#   ../Lib/Interpolation/Bezier/Core
import Lib.Interpolation.Bezier.Core as Bezier

import matplotlib.patches as pat
import Lib.Transformation.Utilities.Mathematics as Mathematics

#P = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.8]])
# L = 18.355664405651
P = np.array([[0, 0], [1, 1], [2, -1], [3, -3], [4, 2], [5, 1]])
# np.array([[0, 0], [1, 1], [2, -1], [3, 0], [4, 2], [5, 1]])
# L = 18.284271247463
#P = np.array([[0, 0], [10, -10], [0, -10], [10, 0]])
# L = 26.791246264404
#P = np.array([[0, 0], [20, 20], [-10, -10], [10, 10]])

# return b and a / b or 0
import time

Bezier_0 = Bezier.Bezier_Cls('Explicit', P, 100)
B_t_0 = Bezier_0.Interpolate()

"""
Bezier_0 = Bezier.Bezier_Cls('Explicit', P, 1000)
t_0 = time.time()
B_t_0 = Bezier_0.Interpolate()
print(time.time() - t_0)

t_1 = time.time()
B_t_dot_0 = Bezier_0.Derivative_1st()
print(time.time() - t_1)
"""

x_min_P = np.min(P[:, 0]); x_max_P = np.max(P[:, 0])
y_min_P = np.min(P[:, 1]); y_max_P = np.max(P[:, 1])

#print(x_min_P,x_max_P)
#print(y_min_P,y_max_P)

min = np.zeros(Bezier_0.dim, dtype=np.float32); max = min.copy()
for i, B_T in enumerate(B_t_0.T):
    min[i] = Mathematics.Min(B_T)[1]
    max[i] = Mathematics.Max(B_T)[1]

#print(min, max)
x_min_B = np.min(B_t_0[:, 0]); x_max_B = np.max(B_t_0[:, 0])
y_min_B = np.min(B_t_0[:, 1]); y_max_B = np.max(B_t_0[:, 1])

print(x_min_B,x_max_B)
print(y_min_B,y_max_B)

#print(Bezier_0.P[0])
print(Bezier_0.Get_Bounding_Box_Parameters())

"""
_, axis = plt.subplots()
plt.plot(P[:, 0], P[:, 1], "--s", label="control points")
plt.plot(B_t_0[:, 0], B_t_0[:, 1], "-", label="B-spline 2")

rec2 = pat.Rectangle(xy = (x_min_B, y_min_B), width = x_max_B - x_min_B, height = y_max_B - y_min_B, facecolor = 'none', edgecolor = 'green', linewidth = 2)
axis.add_patch(rec2)

res = Bezier_0.Get_Bounding_Box_Parameters()
x_min_cls = res[0][0]; x_max_cls = res[1][0]
y_min_cls = res[0][1]; y_max_cls = res[1][1]

rec3 = pat.Rectangle(xy = (x_min_cls, y_min_cls), width = x_max_cls - x_min_cls, height = y_max_cls - y_min_cls, facecolor = 'none', edgecolor = 'blue', linewidth = 2)
axis.add_patch(rec3)

axis.grid()
plt.show()

"""

"""
# [0. 0.5]
# [0.66666667, 0.0] -> this is not a result, but the time {t}!!!
#print(np.round(B_t_0, 2))

#print(Bezier_0.Get_Arc_Length(B_t_dot_0))
"""



