import numpy as np
# System (Default)
import sys
#   Add access if it is not in the system path.
if '../' + 'src' not in sys.path:
    sys.path.append('../..')
import Lib.Interpolation.Utilities as Utilities

def uniform_spaced(n):
    parameters = np.linspace(0, 1, n)
    return parameters

def chord_length(n, P):
    parameters = np.zeros((1, n))
    for i in range(1, n):
        dis = 0
        for j in range(len(P)):
            dis = dis + (P[j][i] - P[j][i-1])**2
        dis = np.sqrt(dis)
        #print(dis)
        parameters[0][i] = parameters[0][i-1] + dis

    for i in range(1, n):
        parameters[0][i] = parameters[0][i]/parameters[0][n-1]
    return parameters[0]

def centripetal(n, P):
    a = 0.5
    parameters = np.zeros((1, n))
    for i in range(1, n):
        dis = 0
        for j in range(len(P)):
            dis = dis + (P[j][i] - P[j][i-1])**2
        dis = np.sqrt(dis)
        parameters[0][i] = parameters[0][i-1] + np.power(dis, a)
    
    for i in range(1, n):
        parameters[0][i] = parameters[0][i] / parameters[0][n-1]
    return parameters[0]

def knot_vector(param, k, N):
    m = N + k
    knot = np.zeros((1, m+1))
    for i in range(k + 1):
        knot[0][i] = 0
    for i in range(m - k, m + 1):
        knot[0][i] = 1
    for i in range(k + 1, m - k):
        for j in range(i - k, i):
            knot[0][i] = knot[0][i] + param[j]
        knot[0][i] = knot[0][i] / k
    return knot[0]

D_X = [1, 1, 0, -0.5, 1, 3]
D_Y = [0, 1, 2,    3, 1, 1]
D = [D_X, D_Y]
N = len(D_X)
k = 2

parameter_1 = chord_length(N, D)
#parameter_1 = uniform_spaced(N)
knot_1 = knot_vector(parameter_1, k, N)
print(knot_1)

p = np.array([[ 1.,  0.], [ 1.,  1.], [ 0.,  2.], [ -0.5,   3.], [ 1.,   1.], [ 3.,   1.]])

knot_10 = Utilities.Generate_Knot_Vector(k, p, 'Chord-Length')
print(knot_10)
