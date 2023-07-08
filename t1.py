import numpy as np

def __Uniformly_Spaced_Method(k, N):
    return np.linspace(0, 1, N + k + 1)

def __Open_Uniform_Method(k, N):
    """
    Description:
        ...

        k .. Degree.
        N .. Number of control points.
    """

    # Get the number of knots.
    n_knots = N + k + 1

    knot_vector = np.zeros(n_knots); knot_spacing = 1.0/(N - k)
    for i in range(n_knots):
        if i < k + 1:
            knot_vector[i] = 0.0
        elif i >= N:
            knot_vector[i] = 1.0
        else:
            knot_vector[i] = (i - k) * knot_spacing

    return knot_vector

def __Centripetal_Method(k, N):
    pass

def __Chord_Length_Method(k, N):
    pass

k = 3
N = 6
knot_1 = __Open_Uniform_Method(k, N)
print(knot_1)

#print(__Uniformly_Spaced_Method(k, N))


