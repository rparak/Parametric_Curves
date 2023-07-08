import numpy as np


def uniform_spaced(n):
    '''
    Calculate parameters using the uniform spaced method.
    :param n: the number of the data points
    :return: parameters
    '''
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

def Generate_Chord_Length_Knot_Vector(k, P):
    num_control_points = len(P)
    chord_lengths = np.zeros(num_control_points)
    
    # Calculate the chord lengths
    L = 0.0
    for i in range(1, num_control_points):
        L_k = np.linalg.norm(P[i] - P[i-1])
        L += L_k
        chord_lengths[i] = chord_lengths[i-1] + L_k

    normalized_lengths = chord_lengths / L
    print(normalized_lengths)
    n_knots = N + k + 1
    knot_vector = np.zeros(n_knots)
    for i in range(n_knots):
        if i < k + 1:
            knot_vector[i] = 0.0
        elif i >= N:
            knot_vector[i] = 1.0
        else:
            knot_vector[i] = np.sum(normalized_lengths[i-k:i]) / k

    return knot_vector

def Generate_Centripetal_Knot_Vector(k, P):
    num_control_points = len(P)
    chord_lengths = np.zeros(num_control_points)
    
    # Calculate the chord lengths
    L = 0.0
    for i in range(1, num_control_points):
        L_k = np.linalg.norm(P[i] - P[i-1])
        L_k = L_k ** 0.5
        L += L_k
        chord_lengths[i] = chord_lengths[i-1] + L_k

    normalized_lengths = chord_lengths / L
    print(normalized_lengths)

    n_knots = N + k + 1
    knot_vector = np.zeros(n_knots)
    for i in range(n_knots):
        if i < k + 1:
            knot_vector[i] = 0.0
        elif i >= N:
            knot_vector[i] = 1.0
        else:
            knot_vector[i] = np.sum(normalized_lengths[i-k:i]) / k

    return knot_vector

D_X = [1, 1, 0, -0.5, 1, 3]
D_Y = [0, 1, 2,    3, 1, 1]
D = [D_X, D_Y]
N = len(D_X)
k = 3

#print(k, N)
p_1 = centripetal(N, D)
print(p_1)

knot = knot_vector(p_1, k, N)
print(knot)

p = np.array([[ 1.,  0.], [ 1.,  1.], [ 0.,  2.], [ -0.5,   3.], [ 1.,   1.], [ 3.,   1.]])
knot2 = Generate_Centripetal_Knot_Vector(k, p)
print(knot2)