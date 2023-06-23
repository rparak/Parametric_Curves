# System (Default)
import sys
#   Add access if it is not in the system path.
sys.path.append('../..')
# Sympy (Symbolic mathematics) [pip3 install sympy]
import sympy as sp
# Custom Script:
#   ../Interpolation/Utilities
import Lib.Interpolation.Utilities as Utilities
#   ../Lib/Transformation/Utilities/Mathematics
import Lib.Transformation.Utilities.Mathematics as Mathematics

class Bezier_Cls(object):
    def __init__(self, P):
        # ...
        self.__t = sp.symbols('t')
        self.__P = P
        self.__B = sp.Float(0.0)
        self.__B_dot = sp.Float(0.0)

        # Degree of a polynomial.
        self.__n = self.__P.shape[0] - 1

    def __C(self, j):
        left_part = 1.0
        if j > 0:
            for m in range(0, j):
                left_part *= self.__n - m
        else:
            m = 0.0

        right_part = 0.0
        for i in range(0, j + 1):
            right_part += (((-1) ** (i + j)) * self.__P[i]) / (Mathematics.Factorial(i)*Mathematics.Factorial(j - i))

        return sp.simplify(left_part * right_part)

    def Derivative_1st(self, method: str):
        try:
            assert method in ['Explicit', 'Polynomial']

            self.__B_dot = sp.Float(0.0)

            if method == 'Explicit':
                # ...
                n = self.__n - 1
                
                # ...
                for i, (p_ii, p_i) in enumerate(zip(self.__P[1:], self.__P)):
                    self.__B_dot += Utilities.Bernstein_Polynomial(i, n, self.__t) * (p_ii - p_i)

                self.__B_dot = self.__n * self.__B_dot

            elif method == 'Polynomial':
                for j in range(0, self.__n + 1):
                    self.__B_dot += self.__t ** (j - 1) * self.__C(j) * j

            return str(sp.Array([self.__B_dot])).replace('1.0*', '')

        except AssertionError as error:
            print(f'[ERROR] Information: {error}')
            print(f'[ERROR] Incorrect type of function input parameters. The calculation method must correspond to \
                  the name Explicit or Polynomial, not {method}.')
    
    def Interpolate(self, method: str):
        try:
            assert method in ['Explicit', 'Polynomial']

            self.__B = sp.Float(0.0)
            
            if method == 'Explicit':
                # ...
                for i, p_i in enumerate(self.__P):
                    self.__B += Utilities.Bernstein_Polynomial(i, self.__n, self.__t) * p_i

            elif method == 'Polynomial':
                # ...
                for j in range(0, self.__n + 1):
                    print(self.__C(j))
                    self.__B += (self.__t ** j) * self.__C(j)

            return str(sp.Array([self.__B])).replace('1.0*', '')

        except AssertionError as error:
            print(f'[ERROR] Information: {error}')
            print(f'[ERROR] Incorrect type of function input parameters. The calculation method must correspond to \
                  the name Explicit or Polynomial, not {method}.')
            
def main():
    P = sp.Array([sp.symbols(f'P_{i}') for i in range(4)])

    Bezier_0 = Bezier_Cls(P)
    B_t_0 = Bezier_0.Interpolate('Explicit')
    print(B_t_0)
    B_t_1 = Bezier_0.Interpolate('Polynomial')
    print(B_t_1)

    B_dot_t_0 = Bezier_0.Derivative_1st('Explicit')
    print(B_dot_t_0)
    B_dot_t_1 = Bezier_0.Derivative_1st('Polynomial')
    print(B_dot_t_1)

if __name__ == '__main__':
    main()