import numpy as np

def Uniformly_Spaced_Method(k, N):
    # open-uniform
    num_knots = N + k + 1

    x = np.zeros(num_knots)
    for i in range(num_knots):
        if i <= k:
            x[i] = 0.0
        elif i <= N:
            x[i] = (i - k)/(N - k)
        elif i <= N + k:
            x[i] = 1.0

    return x

def Centripetal_Method(k, N):
    pass

def Chord_Length_Method(k, N):
    pass

k = 3
N = 100
knot_1 = Uniformly_Spaced_Method(k, N)


