# Numpy (Array computing) [pip3 install numpy]
import numpy as np
# Typing (Support for type hints)
import typing as tp
# Custom Script:
#   ../Interpolation/Utilities
import Lib.Interpolation.Utilities as Utilities
#   ../Lib/Transformation/Utilities/Mathematics
import Lib.Transformation.Utilities.Mathematics as Mathematics

# https://stackoverflow.com/questions/34803197/fast-b-spline-algorithm-with-numpy-scipy
# https://github.com/johntfoster/bspline/blob/master/bspline/splinelab.py
# https://github.com/XuejiaoYuan/BSpline/blob/master/parameter_selection.py

# https://github.com/bhagath555/Symbolic_BSpline/blob/main/Jupyter%20Version/B_Spline.ipynb

# https://github.com/pkicki/cnp-b/tree/master/utils

# https://github.com/kentamt/b_spline

#https://tiborstanko.sk/teaching/geo-num-2017/tp3.html
#https://en.wikipedia.org/wiki/De_Boor%27s_algorithm
#https://cran.r-project.org/web/packages/crs/vignettes/spline_primer.pdf
#https://www.uio.no/studier/emner/matnat/ifi/nedlagte-emner/INF-MAT5340/v05/undervisningsmateriale/kap2-new.pdf

# Book ... Nurbs Book

class B_Spline_Cls(object):
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
        
    def __init__(self, n: int,  method: str, P: tp.List[tp.List[float]], N: int) -> None:
        try:
            assert n < P.shape[0]

            self.__method = method

            # Generate a normalized vector of knots from the selected parameters 
            # using the chosen method.
            self.__t = Utilities.Generate_Knot_Vector(n, P, self.__method)

            # The value of the time must be within the interval of the knot vector: 
            #   t[0] <= Time <= t[-1]
            self.__Time = np.linspace(self.__t[0], self.__t[-1], N)

            # ...
            self.__n = n
            self.__N = N
            self.__P = np.array(P, dtype=np.float32)
            self.__dim = self.__P.shape[1]
            self.__S = np.zeros((N, self.__dim), dtype=np.float32)
            self.__S_dot = np.zeros((N, self.__dim), dtype=np.float32)

        except AssertionError as error:
            print(f'[ERROR] Information: {error}')
            print('[ERROR] Incorrect type of class input parameters.')
            # Error information.
            if n >= P.shape[0]:
                print(f'[ERROR] The degree (n = {n}) of the B-spline must be less than the number of input control points (N = {P.shape[0]}).')
                

    @property
    def n(self) -> int:
        """
        Description:
           ...
        
        Returns:
            (1) ...
        """
                
        return self.__n

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

            # Generate a normalized vector of knots from the selected parameters 
            # using the chosen method.
            self.__t = Utilities.Generate_Knot_Vector(self.__n, P, self.__method)

            # The value of the time must be within the interval of the knot vector: 
            #   t[0] <= Time <= t[-1]
            self.__Time = np.linspace(self.__t[0], self.__t[-1], self.__N)

            # ...
            self.__P = np.array(P, dtype=np.float32)

        except AssertionError as error:
            print(f'[ERROR] Information: {error}')
    
    @property
    def S(self) -> tp.List[tp.List[float]]:
        """
        Description:
           ...
        
        Returns:
            (1) ...
        """
                
        return self.__S
    
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
    def Time(self) -> tp.List[float]:
        """
        Description:
           ...
        
        Returns:
            (1) ...
        """
                
        return self.__Time
    
    @property
    def N(self) -> int:
        """
        Description:
           ...
        
        Returns:
            (1) ...
        """
                
        return self.__Time.shape[0]
    
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
                
        _ = self.Derivative_1st()

        # https://bezier.readthedocs.io/en/stable/python/reference/bezier.curve.html
        L = 0.0
        for _, S_dot_i in enumerate(self.__S_dot):
            L += Mathematics.Euclidean_Norm(S_dot_i)

        return L / (self.N)

    def Optimize_Control_Points(self, N: int) -> tp.List[float]:
        # Least-Squares Fitting of Data with B-Spline Curves
        # https://www.geometrictools.com/Documentation/BSplineCurveLeastSquaresFit.pdf
        # https://github.com/kentamt/b_spline/blob/master/Least_Square_B-spline.ipynb

        try:
            assert N < self.__P.shape[0] and N > self.__n and N != 1

            Time = np.linspace(self.__t[0], self.__t[-1], self.__P.shape[0])

            A = np.zeros((self.__P.shape[0], N), dtype=self.__P.dtype)
            t = Utilities.Generate_Knot_Vector(self.__n, np.zeros((N, 1), dtype=self.__P.dtype), 
                                            'Uniformly-Spaced')
            for i in range(self.__P.shape[0]):
                for j in range(N):
                    A[i, j] = Utilities.Basic_Function(j, self.__n, t, Time[i])

            # estimated control points
            Q = (np.linalg.inv(A.T @ A) @ A.T) @ self.__P
            Q[0] = self.__P[0]; Q[-1] = self.__P[-1]

            return self.__class__(self.__n, self.__method, Q, self.N)

        except AssertionError as error:
            print(f'[ERROR] Information: {error}')
            print('[ERROR] Incorrect type of class input parameters.')
            # Error information.
            if N >= self.__P.shape[0]:
                print(f'[ERROR] The number of optimized control points (N = {N}) must be less than the number of input control points (N_in = {self.__P.shape[0]}).')
            if N <= self.__n:
                print(f'[ERROR] The number of optimized control points (N = {N}) must be greater than the degree (n = {self.__n}) of the B-spline.')
            if N == 1:
                print('[ERROR] The number of optimized control points cannot be equal to 1.')
            
    def Get_Bounding_Box_Parameters(self, method: str) -> tp.Tuple[tp.List[float]]:
        """
        Description:
            ....
        
        Args:
            (1) ...

        Returns:
            (1) ...
        """

        try:
            assert method in ['Control-Points', 'Interpolated-Points']

            min = np.zeros(self.__dim, dtype=np.float32); max = min.copy()

            if method == 'Control-Points':
                for i, P_T in enumerate(self.__P.T):
                    min[i] = Mathematics.Min(P_T)[1]
                    max[i] = Mathematics.Max(P_T)[1]
            else:
                for i, S_T in enumerate(self.__S.T):
                    min[i] = Mathematics.Min(S_T)[1]
                    max[i] = Mathematics.Max(S_T)[1]

            if self.__dim == 2:
                return {'x_min': min[0], 'x_max': max[0], 
                        'y_min': min[1], 'y_max': max[1]}
            else:
                return {'x_min': min[0], 'x_max': max[0], 
                        'y_min': min[1], 'y_max': max[1],
                        'z_min': min[2], 'z_max': max[2]}
        
        except AssertionError as error:
            print(f'[ERROR] Information: {error}')
            print(f'[ERROR] Incorrect type of function input parameters. The method must correspond to the name Control-Points or Interpolated-Points, not {method}.')

    def Reduce_Interpolated_Points(self, epsilon: float) -> tp.List[float]:
        """
        Description:
            ....
        
        Args:
            (1) ...

        Returns:
            (1) ...
        """

        return Utilities.RDP_Simplification(self.__S, epsilon)

    def Derivative_1st(self) -> tp.List[tp.List[float]]:
        """
        Description:
            ....
        
        Args:
            (1) ...

        Returns:
            (1) ...
        """
    
        # https://pages.mtu.edu/~shene/COURSES/cs3621/NOTES/spline/B-spline/bspline-derv.html
        
        # ....
        self.__S_dot = np.zeros(self.__S_dot.shape, dtype=self.__S_dot.dtype)
        self.__S_dot[0, :] = self.__P[0, :]

        t_dot = self.__t[1:-1]
        # ...
        for i, t_i in enumerate(self.__Time):
            for j, (p_i, p_ii) in enumerate(zip(self.__P, self.__P[1:])):   
                Q_i = (self.__n / (t_dot[j + self.__n] - t_dot[j])) * (p_ii - p_i)
                self.__S_dot[i, :] += Utilities.Basic_Function(j, self.__n - 1, t_dot, t_i) * Q_i

        return self.__S_dot
    
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
        self.__S = np.zeros(self.__S.shape, dtype=self.__S.dtype)
        self.__S[0, :] = self.__P[0, :]

        # ...
        for i, t_i in enumerate(self.__Time):
            for j, p_i in enumerate(self.__P):
                self.__S[i, :] += Utilities.Basic_Function(j, self.__n, self.__t, t_i) * p_i

        return self.__S