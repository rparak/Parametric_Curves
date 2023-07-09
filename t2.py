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
        if (u[j+k+1] - u[j+1]) != 0:
            w1 = basic_function(u, j+1, k-1, t) * (u[j+k+1] - t) / (u[j+k+1] - u[j+1])
        if (u[j+k] - u[j]) != 0:  
            if j == 5 and k == 3 and t == 1:
                print(basic_function(u, j, k-1, t))
            w2 = basic_function(u, j, k-1, t)   * (t - u[j])     / (u[j+k] - u[j])        
        var = w1 + w2
    return var

def bspline_basis(u, j, k, t):
    #  Cox-de Boor recursion formula.
    if k == 0:
        if u[j] < t <= u[j+1]:
            return 1.0
        else:
            return 0.0
    else:
        denom1 = u[j+k] - u[j]
        denom2 = u[j+k+1] - u[j+1]
        result = 0.0

        if denom1 != 0:
            if j == 5 and k == 3 and t == 1:
                print(bspline_basis(u, j, k-1, t))
            result += (t - u[j]) / denom1 * bspline_basis(u, j, k-1, t)

        if denom2 != 0:
            result += (u[j+k+1] - t) / denom2 * bspline_basis(u, j+1, k-1, t)

        return result

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
        b1 = bspline_basis(u, j, n, t[i])
        if np.round(b1, 3) != np.round(b, 3):
            print(b1, b)
        S[:, i] = S[:, i] + Q[:, j]*b1

"""
for t_i in t:
    for i in range(6):
        basis = bspline_basis(t_i, i, n, u)
        S[:, i] = S[:, i] + Q[:, i]*basis
"""

fig = plt.figure("B-spline curve", figsize = (6, 3))
plt.plot(Q[0, :], Q[1, :], "--s", label="control points")
plt.plot(S[0, :], S[1, :], "-", label="B spline")
plt.legend()
plt.title("B spline")
plt.xlabel("x")
plt.ylabel("y")
plt.show()
