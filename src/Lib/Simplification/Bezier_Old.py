# Numpy (Array computing) [pip3 install numpy]
import numpy as np
# Sympy (Symbolic mathematics) [pip3 install sympy]
import sympy as sp
import platform
# System (Default)
import sys
if platform.system() == 'Windows':
    # Windows Path.
    sys.path.append('..\\..\\..\\..')
else:
    # Linux (Ubuntu) Path.
    sys.path.append('../../../../../' + 'src') 
# Custom Script:
#   Mathematical Utilities
import Lib.Manipulator.Utilities.Mathematics as Mathematics
import Lib.Manipulator.Interpolation.Utilities as Utilities

class N_Degree(object):
    """
    Description:
        Class for efficient solution of N-degree Bézier curve.

        Note:
            A Bézier curve is a parametric curve used in computer graphics and related fields.

    Initialization of the Class:
        Input:
            (1) num_of_samples [INT]: Number of samples to generate. Must be non-negative.

    Example:
        Initialization:
            Cls = N_Degree(num_of_samples)
        Calculation:
            res = Cls.Solve(points)
        
            where p is equal to [[px_id_0, py_id_0], .., [px_id_n, py_id_n]] in 2D space 
            and [[px_id_0, py_id_0, pz_id_0], .., [px_id_n, py_id_n, pz_id_n]] in 3D space
    """

    def __init__(self, t):
        # << PUBLIC >> #
        # Return evenly spaced numbers over a specified interval.
        self.t = t

    def Solve_Der(self, points):
        # https://pages.mtu.edu/~shene/COURSES/cs3621/NOTES/spline/Bezier/bezier-der.html#:~:text=Therefore%2C%20the%20derivative%20of%20C,of%20the%20original%20Bézier%20curve.
        result = 0.0

        for i in range(0, len(points) - 1):
            # The sum of all positions for the resulting Bézier curve
            result += Utilities.Bernstein_Polynomial(i, len(points) - 2, self.t) * (points[i + 1] - points[i])

        return sp.Array([(len(points) - 1) * result])

    def Solve_Polynomial(self, points):
        n = len(points) - 1

        c = []
        for j in range(0, n + 1):
            pi_var = 1
            for m in range(0, j):
                pi_var *= n - m

            sigma = 0
            for i in range(0, j + 1):
                sigma += (((-1) ** (i + j)) * points[i]) / (Mathematics.Factorial(i)*Mathematics.Factorial(j - i))

            c.append(pi_var * sigma)

        return c

    def Polynomial(self, points):
        result = 0.0
        n = len(points) - 1

        C_j = self.Solve_Polynomial(points)
        for j in range(0, n + 1):
            result += self.t ** j * C_j[j]

        return sp.Array([result])

    def Solve_Der_1st_Recursion(self, points):
        """
        n = len(points) - 1
        result = 0.0
        for i in range(0, n):
            result += Mathematics.Bernstein_Polynomial(i, n - 1, self.t) * (points[i + 1] - points[i])

        return sp.Array([(n) * result])
        """
        # x ^ n = n  * x ^ n - 1
        result = 0.0
        n = len(points) - 1

        C_j = self.Solve_Polynomial(points)
        print(f'C_j = {C_j}')

        coeff = [i*C_ji for i, C_ji in enumerate(C_j)]
        
        print(coeff)
        
        #print(f'Poly = {self.Polynomial(points)}')
        #print(f'Diff_m1 = {(sp.diff(self.Polynomial(points), self.t)[0])}')
        #print(f'Diff_m2 = {sp.expand(sp.diff(self.Solve(points), self.t)[0])}')

        for j in range(0, n + 1):
            result += self.t ** (j - 1) * C_j[j] * j

        return sp.Array([result])

    def Solve(self, points):
        """
        Description: 
            The main control function for creating a Bézier curve of degree n.

        Returns:
            (1) parameter [{0 .. Number of dimensions - 1}] [Int/Float Matrix]: Resulting points of the curve.
        """
        
        result = 0.0
        for i in range(0, len(points)):
            # The sum of all positions for the resulting Bézier curve
            result += Utilities.Bernstein_Polynomial(i, len(points) - 1, self.t) * points[i]

        return sp.Array([result])

# https://fabricesalvaire.github.io/Patro/resources/geometry/bezier.html
# https://pomax.github.io/bezierinfo/
# http://math.aalto.fi/~ahniemi/hss2012/Notes06.pdf
# https://en.wikipedia.org/wiki/De_Casteljau%27s_algorithm
# https://en.wikipedia.org/wiki/Bézier_curve
# find bezier bounding box

# IMPORTANT: add Polynomial form
def main():
    p = [sp.symbols(f'P_{i}') for i in range(4)]
    t = sp.symbols('t')

    Bezier_Ndeg = N_Degree(t)
    # n = 2
    # explicit form: (1 - t)*p0 + t*p1;
    # polynomial form: p0 + t*(p1 - p0);
    # n = 2
    # explicit form: (1 - t)*(p0 + t*(p1 - p0)) + t*(p1 + t*(p2 - p1))
    # polynomial form: (1 - t)*(1 - t)*p0 + 2*(1 - t)*t*p1 + t*t*p2
    #print(Bezier_Ndeg.Solve(p))
    #print(Bezier_Ndeg.Polynomial(p))
    print(Bezier_Ndeg.Solve_Der_1st_Recursion(p))
    #print(Bezier_Ndeg.Solve_Der(p))

if __name__ == '__main__':
    main()
