import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# https://github.com/kentamt/b_spline/blob/master/B-spline.ipynb
def open_uniform_vector(m, degree):
    u = np.zeros((m, 1), dtype=float)
    j = 1
    for i in range(m):
        if i <= degree: 
            u[i] = 0.0
        elif i < m - (degree+1): 
            u[i] = 1.0 / (m - 2*(degree+1) + 1) * j
            j += 1
        else: 
            u[i] = 1.0 
    return u.flatten()

def basic_function(u, j, k, t):   
    w1 = 0.0
    w2 = 0.0
    if k == 0: 
        if u[j] < t and t <= u[j+1]:
            var = 1.0
        else:
            var = 0.0
    else:
        if (u[j+k+1]-u[j+1]) != 0:
            w1 = basic_function(u, j+1, k-1, t) * (u[j+k+1] - t) / (u[j+k+1] - u[j+1])
        if (u[j+k]-u[j]) != 0:  
            w2 = basic_function(u, j, k-1, t)   * (t - u[j])     / (u[j+k] - u[j])        
        var = w1 + w2
    return var

def de_boor(control_points, knots, t, degree):
    n = len(control_points) - 1
    k = degree + 1

    # Find the interval [u_i, u_i+1] where t lies
    i = 0
    while i < n + k and knots[i + 1] <= t:
        i += 1

    # Initialize the control points for the current level
    d = control_points[i - k:i + 1].copy()

    # Apply de Boor's algorithm
    for r in range(1, k):
        for j in range(i, i - k + r, -1):
            alpha = (t - knots[j]) / (knots[j + k - r] - knots[j])

            d[j] = (1 - alpha) * d[j - 1] + alpha * d[j]

    return d[k - 1]

# Example usage
control_points = np.array([[0, 0], [1, 1], [2, -1], [3, 0], [4, 2], [5, 1]])

#result = de_boor(control_points, knots, t, degree)
#print(f"The point at t={t} is {result}")


p = control_points.shape[0]
n = 3
m = p + n + 1

u = open_uniform_vector(m, n)
t = np.linspace(0.0, u[-1], int(u[-1]/0.01))

Q = control_points.T
S = np.zeros((2, len(t)))
S[:, 0] = Q[:, 0]
for i in range(len(t)):
    if i==0:
        continue
        
    for j in range(6):
        b = basic_function(u, j, n, t[i]) 
        S[:, i] = S[:, i] + Q[:, j]*b

fig = plt.figure("B-spline curve", figsize = (6, 3))
plt.plot(Q[0, :], Q[1, :], "--s", label="control points")
plt.plot(S[0, :], S[1, :], "-", label="B spline")
plt.legend()
plt.title("B spline")
plt.xlabel("x")
plt.ylabel("y")
plt.show()