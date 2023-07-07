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

# Book ... Nurbs Book


# Generation of the Knot Vector:
#   Method: 
#       1\ Open Uniform Method
#       2\ Uniform Method
#       3\ Chord Lenght Method
#       4\ Centripetal Method

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
        
    def __init__(self, P: tp.List[tp.List[float]], N: int) -> None:
        self.__P = np.array(P, dtype=np.float32)
        self.__dim = self.__P.shape[1] - 1

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
    
    def Get_Arc_Length(self, B_dot: tp.List[tp.List[float]]) -> float:
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
        for _, B_dot_i in enumerate(B_dot):
            L += Mathematics.Euclidean_Norm(B_dot_i)

        return L / self.N

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
                
        # ...
        min = np.zeros(self.__dim, dtype=np.float32); max = min.copy()

        for i, B_T in enumerate(self.__B.T):
            min[i] = Mathematics.Min(B_T)[1]
            max[i] = Mathematics.Max(B_T)[1]

        return (min, max)

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

        return self.__B