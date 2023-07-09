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
        for _, S_dot_i in enumerate(self.__S_dot):
            L += Mathematics.Euclidean_Norm(S_dot_i)

        return L / self.N

    def Simplify(self, epsilon: float) -> tp.List[float]:
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
                
        # ....
        self.__S_dot = np.zeros(self.__S_dot.shape, dtype=self.__S_dot.dtype)
            
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

        return self.__S