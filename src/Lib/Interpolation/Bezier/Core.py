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
            (1) method [string]: The name of the method to calculate the interpolation function.
                                    Note:
                                        method = 'Explicit' or 'Polynomial'.
            (2) P [Vector<float> mxn]: Input control points to be interpolated.
                                          Note:
                                            Where m is the number of points and n is the dimension (2-D, 3-D).
            (3) N [int]: The number of points to be generated in the interpolation function.

        Example:
            Initialization:
                # Assignment of the variables.
                method = 'Explicit'; N = 100
                P = np.array([[1.00,  0.00], 
                              [2.00, -0.75], 
                              [3.00, -2.50], 
                              [3.75, -1.25], 
                              [4.00,  0.75], 
                              [5.00,  1.00]], dtype=np.float32)

                # Initialization of the class.
                Cls = Bezier_Cls(method, P, N)

            Features:
                # Properties of the class.
                Cls.P; Cls.Time, Cls.N
                ...
                Cls.dim

                # Functions of the class.
                Cls.Get_Arc_Length()
                ...
                Cls.Interpolate()
    """
        
    def __init__(self, method: str, P: tp.List[tp.List[float]], N: int) -> None:
        try:
            assert method in ['Explicit', 'Polynomial']

            # Convert the string to an identification number.
            #   'Explicit': 0; 'Polynomial': 1
            self.__method_id = 0 if method == 'Explicit' else 1

            # The value of the time must be within the interval of the knot vector: 
            #   t[0] <= Time <= t[-1]
            self.__Time = np.linspace(Utilities.CONST_T_0, Utilities.CONST_T_1, N)

            # Initialization of other class parameters.
            #   Control Points.
            self.__P = np.array(P, dtype=np.float32)
            #   Dimension (2-D, 3-D).
            self.__dim = self.__P.shape[1]
            #   Interpolated points.
            self.__B = np.zeros((N, self.__dim), dtype=np.float32)
            #   First derivation of interpolated points.
            self.__B_dot = np.zeros((N, self.__dim), dtype=np.float32)
            #   Degree of a polynomial.
            self.__n = self.__P.shape[0] - 1

        except AssertionError as error:
            print(f'[ERROR] Information: {error}')
            print(f'[ERROR] Incorrect type of class input parameters. The calculation method must correspond to the name Explicit or Polynomial, not {method}.')

    @property
    def P(self) -> tp.List[tp.List[float]]:
        """
        Description:
           Get the control points of the curve.
        
        Returns:
            (1) parameter [Vector<float> mxn]: Control points.
                                                Note:
                                                    Where m is the number of points and n is the dimension (2-D, 3-D).
        """
                
        return self.__P
    
    @P.setter
    def P(self, P: tp.List[tp.List[float]]) -> None:
        """
        Description:
           Set the new control points of the curve.
        
        Args:
            (1) P [Vector<float> mxn]: Control points.
                                        Note:
                                            Where m is the number of points and n is the dimension (2-D, 3-D).
        """
                
        try:
            assert P.shape[1] == self.__dim

            self.__P = np.array(P, dtype=np.float32)

        except AssertionError as error:
            print(f'[ERROR] Information: {error}')
            print(f'[ERROR] Incorrect dimensions of input control points. The point dimension must be {self.__dim} and not {P.shape[1]}.')
    
    @property
    def B(self) -> tp.List[tp.List[float]]:
        """
        Description:
           Get the interpolated points of the curve.
        
        Returns:
            (1) parameter [Vector<float> mxn]: Interpolated points.
                                                Note:
                                                    Where m is the number of points and n is the dimension (2-D, 3-D).
        """
                
        return self.__B
    
    @property
    def Time(self) -> tp.List[float]:
        """
        Description:
           Get the time as an interval of values from 0 to 1.
        
        Returns:
            (1) parameter [Vector<float> 1xn]: Time.
                                                Note:
                                                    Where n is the number of points.
        """
                
        return self.__Time
    
    @property
    def N(self) -> int:
        """
        Description:
           Get the number of points to be generated in the interpolation function.
        
        Returns:
            (1) parameter [int]: Number of interpolated points. 
        """
                
        return self.__Time.shape[0]
    
    @property
    def dim(self) -> int:
        """
        Description:
           Get the dimension (2-D, 3-D) of the control/interpolated points.
        
        Returns:
            (1) parameter [int]: The dimension of the points at which the interpolation is performed.
        """
                
        return self.__dim
    
    def Get_Arc_Length(self) -> float:
        """
        Description:
            Obtain the arc length L(t) of the general parametric curve.

            The arc length L(t) is defined by:
                
                L(t) = \int_{0}^{t} ||B'(t)||_{2} dt.

        Returns:
            (1) parameter [float]: The arc length L(t) of the general parametric curve.
        """
                
        # ...
        _ = self.Derivative_1st()
        
        L = 0.0
        for _, B_dot_i in enumerate(self.__B_dot):
            L += Mathematics.Euclidean_Norm(B_dot_i)

        return L / self.N
    
    def __Get_Initial_Min_Max_BB(self, P_0: tp.List[float], P_N: tp.List[float]) -> tp.Tuple[tp.List[float], 
                                                                                             tp.List[float]]:
        """
        Description:
           Obtain the minimum and maximum values in each axis from the input control points (P_0, P_N).
        
        Args:
            (1, 2) P_0, P_N [Vector<float> mxn]: First (P_0) and last (P_N) control point.
                                                    Note:
                                                        Where m is the number of points and n is the dimension (2-D, 3-D).

        Returns:
            (1) parameter [Vector<float> 1xn]: Minimum values in each axis from control points P_0 and P_N.
            (2) parameter [Vector<float> 1xn]: Maximum values in each axis from control points P_0 and P_N.
                                                Note:
                                                    Where n is the dimension (2-D, 3-D).
        """

        min = np.zeros(self.__dim, dtype=np.float32); max = min.copy()
        for i, (p_0_i, p_n_i) in enumerate(zip(P_0, P_N)):
            min[i] = Mathematics.Min([p_0_i, p_n_i])[1]
            max[i] = Mathematics.Max([p_0_i, p_n_i])[1]

        return (min, max)
    
    def __Get_B_t(self, P: tp.List[float], t: tp.List[float]) -> tp.List[float]:
        """
        Description:
            Obtain the interpolated control points with the defined time value t.
        
        Args:
            (1) P [Vector<float> mxn]: Input control points to be interpolated.
                                        Note:
                                            Where m is the number of points and n is the dimension (2-D, 3-D).
            (2) t [Vector<float> 1xk]: Defined time value t.
                                        Note:
                                            Where k is the number of values in the vector.

        Returns:
            (1) parameter [Vector<float> 1xn]: Interpolated points.
                                                Note:
                                                    Where n is the dimension (2-D, 3-D).
        """

        B = np.zeros(self.__dim, dtype=np.float32)
        for j, p_j in enumerate(P):
            B += Utilities.Bernstein_Polynomial(j, self.__n, t) * p_j

        return B

    def __C(self, j: int) -> tp.List[float]:
        """
        Description:
            ....
        
        Args:
            (1) j [int]: ...

        Returns:
            (1) parameter [Vector<float> ..]:
        """
                
        eq_ls = 1.0

        for m in range(0, j):
            eq_ls *= self.__n - m

        eq_rs = 0.0
        for i in range(0, j + 1):
            eq_rs += (((-1) ** (i + j)) * self.__P[i]) / (Mathematics.Factorial(i)*Mathematics.Factorial(j - i))

        return eq_ls * eq_rs
    
    def Get_Bounding_Box_Parameters(self, limitation: str) -> tp.Tuple[tp.List[float]]:
        """
        Description:
            Obtain the bounding parameters (min, max) of the general parametric 
            curve (interpolated points) as well as the control points.
        
        Args:
            (1) limitation [string]: The limitation to be used to describe the bounding box.
                                        Note:
                                            limitation = 'Control-Points'
                                                - The result depends on the parameters (min, max) of the control points.
                                            limitation = 'Interpolated-Points'
                                                - The result depends on the parameters (min, max) of the general parametric 
                                                curve (interpolated points).

        Returns:
            (1) parameter [Dictionary {'x_min': int, 'y_min': int, etc.}]: Bounding box parameters (min, max) defined by the limitation 
                                                                           from the arguments of the function.
                                                                            Note:
                                                                                The number of values in both parameters min and max depends 
                                                                                on the dimension of the points.
        """
        
        try:
            assert limitation in ['Control-Points', 'Interpolated-Points']
        
            if limitation == 'Control-Points':
                min = np.zeros(self.__dim, dtype=np.float32); max = min.copy()
                for i, P_T in enumerate(self.__P.T):
                    min[i] = Mathematics.Min(P_T)[1]
                    max[i] = Mathematics.Max(P_T)[1]
            else:
                # https://snoozetime.github.io/2018/05/22/bezier-curve-bounding-box.html

                # Obtain the minimum and maximum values in each axis from 
                # the input control points (P_0, P_N).
                (min, max) = self.__Get_Initial_Min_Max_BB(self.__P[0], self.__P[-1])

                # ...
                coeff = np.array([i*self.__C(i) for i in range(1, self.__n + 1)],
                                 dtype=np.float32).T
                
                # ....
                for i, coeff_i in enumerate(coeff):
                    if coeff_i.size != 1:
                        roots = Mathematics.Roots(coeff_i[::-1])

                        # The value of the time must be within the interval of the knot vector: 
                        #   t[0] <= Time <= t[-1]
                        t_tmp = []
                        for _, roots_i in enumerate(roots):
                            if Utilities.CONST_T_0 <= roots_i <= Utilities.CONST_T_1:
                                t_tmp.append(roots_i)
                        
                        # Remove duplicates from the vector.
                        t_tmp = np.array([*set(t_tmp)], dtype=np.float32)
                        
                        if t_tmp.size == 0:
                            continue
                        else:
                            t = t_tmp
                    else:
                        if Utilities.CONST_T_0 <= coeff_i <= Utilities.CONST_T_1:
                            t = np.array(coeff_i, dtype=np.float32)
                        else:
                            continue

                    # ...
                    B_i = []
                    for _, t_i in enumerate(t):
                        B_i.append(self.__Get_B_t(self.__P[:, i], np.array(t_i, dtype=np.float32)))
                    B_i = np.array(B_i, dtype=np.float32).flatten()

                    # ...
                    min[i] = Mathematics.Min(np.append(min[i], B_i))[1]
                    max[i] = Mathematics.Max(np.append(max[i], B_i))[1]

            if self.__dim == 2:
                return {'x_min': min[0], 'x_max': max[0], 
                        'y_min': min[1], 'y_max': max[1]}
            else:
                return {'x_min': min[0], 'x_max': max[0], 
                        'y_min': min[1], 'y_max': max[1],
                        'z_min': min[2], 'z_max': max[2]}
        
        except AssertionError as error:
            print(f'[ERROR] Information: {error}')
            print(f'[ERROR] Incorrect type of function input parameters. The limitation must correspond to the name Control-Points or Interpolated-Points, not {limitation}.')

    def Reduce_Interpolated_Points(self, epsilon: float) -> tp.List[float]:
        """
        Description:
            A function to simplify (reduce) a given array of interpolated points using 
            the Ramer-Douglas-Peucker algorithm.
        
        Args:
            (1) epsilon [float]: The coefficient determines the similarity between the original a
                                 and the approximated curve. 
                                    Note: 
                                        epsilon > 0.0.

        Returns:
            (1) parameter [Vector<float> ]: Simplified (reduced) vector of interpolated points {B}.
        """

        return Utilities.RDP_Simplification(self.__B, epsilon)
    
    def Derivative_1st(self) -> tp.List[tp.List[float]]:
        """
        Description:
            ....

        Returns:
            (1) parameter [Vector<float> ..]: 
        """
                
        # ....
        self.__B_dot = np.zeros(self.__B_dot.shape, dtype=self.__B_dot.dtype)

        if self.__method_id == 0:
            # ...
            n = self.__n - 1
            
            # ...
            for i, (p_i, p_ii) in enumerate(zip(self.__P, self.__P[1:])):
                for j, (p_ij, p_iij) in enumerate(zip(p_i, p_ii)):
                    self.__B_dot[:, j] += Utilities.Bernstein_Polynomial(i, n, self.__Time) * (p_iij - p_ij)

            self.__B_dot = self.__n * self.__B_dot

        elif self.__method_id == 1:
            for j in range(1, self.__n + 1):
                for i, C_j in enumerate(self.__C(j)):
                    self.__B_dot[:, i] += (self.__Time ** (j - 1)) * C_j * j
            
        return self.__B_dot
    
    def Interpolate(self) -> tp.List[tp.List[float]]:  
        """
        Description:
            ....

        Returns:
            (1) parameter [Vector<float> ..]: 
        """
                  
        # ....
        self.__B = np.zeros(self.__B.shape, dtype=self.__B.dtype)

        if self.__method_id == 0:
            # ...
            for i, p_i in enumerate(self.__P):
                for j, p_ij in enumerate(p_i):
                    self.__B[:, j] += Utilities.Bernstein_Polynomial(i, self.__n, self.__Time) * p_ij

        elif self.__method_id == 1:
            # ...
            for j in range(1, self.__n + 1):
                for i, C_j in enumerate(self.__C(j)): 
                    self.__B[:, i] += (self.__Time ** j) * C_j

        return self.__B