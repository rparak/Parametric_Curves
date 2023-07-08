import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# https://github.com/kentamt/b_spline/blob/master/Least_Square_B-spline.ipynb

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


x = np.arange(0, 10, 0.15)
yt = 10 - 0.2 * x**2 + 2 * np.sin(x)
y  = 10 - 0.1 * x**2 + 2 * np.sin(x) + np.random.normal(0.0, 1.0, len(x))

n = 4 # num of control points
d = 3 # degree of b-spline
nk = d + n + 1 # num of knot 
m = len(x) # num of data points

u = open_uniform_vector(nk, d)
t = np.linspace(0.0, 1, len(x))

# Least Square
P = np.array([x, y]).T # samples
A = np.zeros((m, n)) # Basic function Matrix
for k in range(m):
    for j in range(n):
        A[k, j] = basic_function(u, j, d, t[k])
        
X = np.linalg.inv(A.T.dot(A)).dot(A.T) # X = [A'A]^(-1) A'
Q = X.dot(P) # estimated control points

S = np.zeros((2, len(t)))
Q = Q.T
S[:, 0] = Q[:, 0]
for i in range(len(t)):
    if i==0:
        continue
        
    for j in range(n):
        #print(j)
        b = basic_function(u, j, d, t[i]) 
        print(Q[:, j]*b) 
        S[:, i] = S[:, i] + Q[:, j]*b

fig = plt.figure("LSBspline", figsize=(6, 3))
plt.plot(x, y,"o", c="lightgray", mfc="none",  ms=10, label="Samples")
plt.plot(Q[0, :], Q[1, :], "s--", ms=10,label="Control points")
plt.plot(S[0, :], S[1, :], "-", label="B-spline curve")

plt.title("Least square B-spline fitting")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()      
plt.show()