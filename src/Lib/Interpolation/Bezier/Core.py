# Numpy (Array computing) [pip3 install numpy]
import numpy as np
# Typing (Support for type hints)
import typing as tp
# Custom Script:
#   ../Interpolation/Utilities
import Lib.Interpolation.Utilities as Utilities
#   ../Lib/Transformation/Utilities/Mathematics
import Lib.Transformation.Utilities.Mathematics as Mathematics

class Bezier_Cls(object):
    """
    Description:
        ....

    Initialization of the Class:
        Args:
            (1) ...

        Example:
            Initialization:
                # Assignment of the variables.
                ...

                # Initialization of the class.
                Cls = ...

            Features:
                # Properties of the class.
                Cls..
                ...
                Cls..

                # Functions of the class.
                Cls..
                ...
                Cls..
    """
        
    def __init__(self, method: str, P: tp.List[tp.List[float]], N: int) -> None:
        try:
            assert method in ['Explicit', 'Polynomial']

            self.__method_id = 0 if method == 'Explicit' else 1

            # The time (roots) value must be within the interval: 0.0 <= t <= 1.0
            self.__t = np.linspace(Utilities.CONST_T_0, Utilities.CONST_T_1, N)

            # ...
            self.__P = np.array(P, dtype=np.float32)
            self.__dim = self.__P.shape[1]
            self.__B = np.zeros((N, self.__dim), dtype=np.float32)
            self.__B_dot = np.zeros((N, self.__dim), dtype=np.float32)

            # Degree of a polynomial.
            self.__n = self.__P.shape[0] - 1

        except AssertionError as error:
            print(f'[ERROR] Information: {error}')
            print(f'[ERROR] Incorrect type of function input parameters. The calculation method must correspond to the name Explicit or Polynomial, not {method}.')

    @property
    def P(self) -> tp.List[tp.List[float]]:
        """
        Description:
           ...
        
        Returns:
            (1) ...
        """
                
        return self.__P
    
    @P.setter
    def P(self, P: tp.List[tp.List[float]]) -> None:
        """
        Description:
           ...
        
        Returns:
            (1) ...
        """
                
        try:
            assert P.shape[1] == self.__dim

            self.__P = np.array(P, dtype=np.float32)

        except AssertionError as error:
            print(f'[ERROR] Information: {error}')
    
    @property
    def B(self) -> tp.List[tp.List[float]]:
        """
        Description:
           ...
        
        Returns:
            (1) ...
        """
                
        return self.__B
    
    @property
    def t(self) -> tp.List[float]:
        """
        Description:
           ...
        
        Returns:
            (1) ...
        """
                
        return self.__t
    
    @property
    def N(self) -> int:
        """
        Description:
           ...
        
        Returns:
            (1) ...
        """
                
        return self.__t.shape[0]
    
    @property
    def dim(self) -> int:
        """
        Description:
           ...
        
        Returns:
            (1) ...
        """
                
        return self.__dim
    
    def Get_Arc_Length(self) -> float:
        """
        Description:
           ...
        
        Args:
            (1) ...

        Returns:
            (1) ...
        """
                
        # https://bezier.readthedocs.io/en/stable/python/reference/bezier.curve.html
        L = 0.0
        for _, B_dot_i in enumerate(self.__B_dot):
            L += Mathematics.Euclidean_Norm(B_dot_i)

        return L / self.N
    
    def __Get_Initial_Min_Max_BB(self, P_0: tp.List[float], P_N: tp.List[float]) -> tp.Tuple[tp.List[float], 
                                                                                             tp.List[float]]:
        """
        Description:
           ...
        
        Args:
            (1) ...

        Returns:
            (1) ...
        """

        min = np.zeros(self.__dim, dtype=np.float32); max = min.copy()
        for i, (p_0_i, p_n_i) in enumerate(zip(P_0, P_N)):
            min[i] = np.minimum(p_0_i, p_n_i)
            max[i] = np.maximum(p_0_i, p_n_i)

        return (min, max)
    
    def __Get_B_t(self, P: tp.List[float], t: tp.Union[float, tp.List[float]]) -> tp.List[float]:
        """
        Description:
            Interpolate with defined t value ....
        
        Args:
            (1) ...

        Returns:
            (1) ...
        """

        B = np.zeros(self.__dim, dtype=np.float32)
        for j, p_j in enumerate(P):
            B += Utilities.Bernstein_Polynomial(j, self.__n, t) * p_j

        return B

    def Get_Bounding_Box_Parameters(self) -> tp.Tuple[tp.List[float], 
                                                      tp.List[float]]:
        """
        Description:
            ....
        
        Args:
            (1) ...

        Returns:
            (1) ...
        """
                
        # https://snoozetime.github.io/2018/05/22/bezier-curve-bounding-box.html

        # ...
        (min, max) = self.__Get_Initial_Min_Max_BB(self.__P[0], self.__P[-1])

        # ...
        coeff = np.array([i*self.__C(i) for i in range(1, self.__n + 1)],
                          dtype=np.float32).T
        
        # ....
        for i, coeff_i in enumerate(coeff):
            # Note:
            #   The time (roots) value must be within the interval: 0.0 <= t <= 1.0
            if coeff_i.size != 1:
                roots = Mathematics.Roots(coeff_i[::-1])
                t_tmp = np.array([Mathematics.Clamp(roots_i, Utilities.CONST_T_0, 
                                                    Utilities.CONST_T_1) for _, roots_i in enumerate(roots)], dtype=np.float32)
                print(roots)
                print(t_tmp)
                t = np.array([Mathematics.Min(t_tmp)[1], Mathematics.Max(t_tmp)[1]], dtype=np.float32)
            else:
                t = Mathematics.Clamp(coeff_i[::-1], Utilities.CONST_T_0, Utilities.CONST_T_1)

            # ...
            B_i = self.__Get_B_t(self.__P[:, i], t)

            # ...
            min[i] = Mathematics.Min(np.append(min[i], B_i))[1]
            max[i] = Mathematics.Max(np.append(max[i], B_i))[1]

        return (min, max)
            
    def __C(self, j: int) -> tp.List[float]:
        """
        Description:
            ....
        
        Args:
            (1) ...

        Returns:
            (1) ...
        """
                
        eq_ls = 1.0

        for m in range(0, j):
            eq_ls *= self.__n - m

        eq_rs = 0.0
        for i in range(0, j + 1):
            eq_rs += (((-1) ** (i + j)) * self.__P[i]) / (Mathematics.Factorial(i)*Mathematics.Factorial(j - i))

        return eq_ls * eq_rs

    def Simplify(self, epsilon: float) -> tp.List[float]:
        """
        Description:
            ....
        
        Args:
            (1) ...

        Returns:
            (1) ...
        """

        return Utilities.RDP_Simplification(self.__B, epsilon)
    
    def Derivative_1st(self) -> tp.List[tp.List[float]]:
        """
        Description:
            ....
        
        Args:
            (1) ...

        Returns:
            (1) ...
        """
                
        # ....
        self.__B_dot = np.zeros(self.__B_dot.shape, dtype=self.__B_dot.dtype)

        if self.__method_id == 0:
            # ...
            n = self.__n - 1
            
            # ...
            for i, (p_i, p_ii) in enumerate(zip(self.__P, self.__P[1:])):
                for j, (p_ij, p_iij) in enumerate(zip(p_i, p_ii)):
                    self.__B_dot[:, j] += Utilities.Bernstein_Polynomial(i, n, self.__t) * (p_iij - p_ij)

            self.__B_dot = self.__n * self.__B_dot

        elif self.__method_id == 1:
            for j in range(1, self.__n + 1):
                for i, C_j in enumerate(self.__C(j)):
                    self.__B_dot[:, i] += (self.__t ** (j - 1)) * C_j * j
            
        return self.__B_dot
    
    def Interpolate(self) -> tp.List[tp.List[float]]:  
        """
        Description:
            ....
        
        Args:
            (1) ...

        Returns:
            (1) ...
        """
                  
        # ....
        self.__B = np.zeros(self.__B.shape, dtype=self.__B.dtype)

        if self.__method_id == 0:
            # ...
            for i, p_i in enumerate(self.__P):
                for j, p_ij in enumerate(p_i):
                    self.__B[:, j] += Utilities.Bernstein_Polynomial(i, self.__n, self.__t) * p_ij

        elif self.__method_id == 1:
            # ...
            for j in range(1, self.__n + 1):
                for i, C_j in enumerate(self.__C(j)): 
                    self.__B[:, i] += (self.__t ** j) * C_j

        return self.__B