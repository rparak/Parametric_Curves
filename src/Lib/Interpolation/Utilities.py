# Numpy (Array computing) [pip3 install numpy]
import numpy as np
# Typing (Support for type hints)
import typing as tp
# Custom Script:
#   ../Lib/Transformation/Core
import Lib.Transformation.Core as Transformation
#   ../Lib/Transformation/Utilities/Mathematics
import Lib.Transformation.Utilities.Mathematics as Mathematics

"""
Description:
    Initialization of constants.
"""
# Time t ∈ [0: The starting value of the sequence, 
#           1: The end value of the sequence] {0.0 <= t <= 1.0}
CONST_T_0 = 0.0
CONST_T_1 = 1.0

def Lerp(method: str, p_0: tp.List[float], p_1: tp.List[float], t: float) -> tp.List[float]:
    """
    Description:
        Linear interpolation (Lerp) is a method of curve fitting using linear polynomials to construct new data 
        points over the range of a discrete set of known data points. 

        The equation of the Lerp function is defined as follows:
            1\ Explicit Method:
                B(t) = (1 - t) * p_{0} + t * p_{1},
            
            2\ Polynomial Method:
                B(t) = p_{0} + t * (p_{1} - p_{0}),

            where t can take values 0.0 <= t <= 1.0.
    
    Args:
        (1) method [string]: The name of the method to calculate the {Lerp} function.
                             Note:
                                method = 'Explicit' or 'Polynomial'
        (2) p_0, p_1 [Vector<float> 1xn]: Input points to be interpolated.
                                          Note:
                                            Where n is the number of dimensions of the point.
        (3) t [float]: Time t (0.0 <= t <= 1.0).
        
    Returns: 
        (1) parameter [Vector<float> 1xn]: Interpolated point at time {t}.
                                           Note:
                                            Where n is the number of dimensions of the point.
    """

    try:
        assert method in ['Explicit', 'Polynomial']

        # The time value must be within the interval: 0.0 <= t <= 1.0
        t = Mathematics.Clamp(t, CONST_T_0, CONST_T_1)

        if method == 'Explicit':
            # Equation:
            #   B(t) = (1 - t) * p_{0} + t * p_{1}
            return (1 - t) * p_0 + t * p_1
        elif method == 'Polynomial':
            # Equation:
            #   B(t) = p_{0} + t * (p_{1} - p_{0})
            return p_0 + t * (p_1 - p_0)
        
    except AssertionError as error:
        print(f'[ERROR] Information: {error}')
        print(f'[ERROR] Incorrect type of function input parameters. The calculation method must correspond to \
              the name Explicit or Polynomial, not {method}.')

def Slerp(method: str, q_0: tp.List[float], q_1: tp.List[float], t: float) -> tp.List[float]:
    """
    Description:
        Performs a spherical linear interpolation (Slerp) between the given quaternions 
        and stores the result in this quaternion.

        The equation of the Slerp function is defined as follows:
            1\ Geometric:
                B(t) = (sin[(1 - t) * theta] / sin(theta)) * q_{0} + (sin[t * theta] / sin(theta)) * q_{1},
            2\ Quaternion:
                B(t) = (q_{1} * q_{0}^(-1))^(t) * q_0,

            where t can take values 0.0 <= t <= 1.0.

    Args:
        (1) method [string]: The name of the method to calculate the {Slerp} function.
                             Note:
                                method = 'Geometric' or 'Quaternion'
        (2) q_0, q_1 [Vector<float> 1x4]: Input quaternions to be interpolated.
        (3) t [float]: Time t (0.0 <= t <= 1.0).

    Returns: 
        (1) parameter [Vector<float> 1x4]: Interpolated quaternion at time {t}.                  
    """

    try:
        assert method in ['Geometric', 'Quaternion']
    
        # The time value must be within the interval: 0.0 <= t <= 1.0
        t = Mathematics.Clamp(t, CONST_T_0, CONST_T_1)

        if isinstance(q_0, Transformation.Quaternion_Cls) and isinstance(q_1, Transformation.Quaternion_Cls):
            q_0 = q_0.Normalize()
            q_1 = q_1.Normalize()
        else:
            q_0 = Transformation.Quaternion_Cls(q_0, np.float32).Normalize()
            q_1 = Transformation.Quaternion_Cls(q_1, np.float32).Normalize()

        if method == 'Geometric':
            # Calculate angle between quaternions (q_0, q_1).
            q_01_angle = q_0.Dot(q_1)

            if q_01_angle < 0.0:
                q_0 = -q_0; q_01_angle = -q_01_angle

            if q_01_angle > 0.9995:
                # If the input quaternions are too close, perform a linear 
                # interpolation (Lerp):
                #   Lerp (polynomial form): q_{0} + t * (q_{1} - q_{0})
                return Lerp('Polynomial', q_0, q_1, t)

            # Auxiliary expression for the final equation.
            theta = np.arccos(q_01_angle); sin_theta = np.sin(theta)

            return (np.sin((1 - t) * theta) / sin_theta) * q_0 + (np.sin(t * theta) / sin_theta) * q_1
        elif method == 'Quaternion':
            return (q_1 * q_0.Inverse())**t * q_0
    
    except AssertionError as error:
        print(f'[ERROR] Information: {error}')
        print(f'[ERROR] Incorrect type of function input parameters. The calculation method must correspond to \
              the name Geometric or Quaternion, not {method}.')
        
def Bernstein_Polynomial(i: int, n: int, t: tp.List[float]) -> tp.List[float]:
    """
    Description:
        Bernstein polynomials form the basis of the Bézier elements used in isogeometric analysis.

        For a given {n >= 0}, define the {n + 1} Bernstein basis polynomials of degree {n} on [0,1] as:
            b_{i, n}(t) = (n k) * (1 - t)^(n - i) * (t^i), i = 0,..,n,

            where (n k) is a binomial coefficient.

        There are four of them for n = 3, for example:
            b_{0, 3} = (1 - t)^3
            b_{1, 3} = 3 * t * (1 - t)^2
            b_{2, 3} = 3 * (t^2) * (1 - t)
            b_{3, 3} = t^3

    Args:
        (1) i [int]: Iteration.
        (2) n [int]: Degree of a polynomial.
        (3) t [Vector<float> 1xk]: Time t ∈ [0: The starting value of the sequence, 1: The end value of the sequence].
                                   {0.0 <= t <= 1.0}
                                   Note:
                                    Where k is the number of elements of the time vector.
                             
    Returns:
        (1) parameter [Vector<float> 1xk]: A Bernstein polynomial of degree n.
                                           Note:
                                            Where k is the number of elements of the time vector.
    """

    try:
        assert n >= Mathematics.CONST_NULL
  
        # Bernstein basis polynomials b_{i, n}(t).
        return Mathematics.Binomial_Coefficient(n, i) * (1 - t) ** (n - i) * (t ** i)

    except AssertionError as error:
        print(f'[ERROR] Information: {error}')
        print(f'[ERROR] The input condition for the polynomial calculation is not satisfied. \
              The degree of n ({n}) must be greater than or equal to {Mathematics.CONST_NULL}.')


def Basic_Function():
    # https://tiborstanko.sk/teaching/geo-num-2017/tp3.html
    # https://en.wikipedia.org/wiki/De_Boor%27s_algorithm
    pass

def Simple_Simplification(P: tp.List[float], s_f: int) -> tp.List[float]:
    """
    Description:
        A function to simplify the point vector by a simplification factor s_{f}. The first and end point are unchanged, the others 
        depend on the simplification factor.

        The points must be in the following form:
            P = [p_0{x, y, ..}, p_1{x, y, ..}, ...]

        Example:
            Input Points: 
                P = [1.0, 1.0], [1.25, 2.0], [1.75, 2.0], [2.0, 1.0], [1.0, -1.0], [1.25, -2.0], [1.75, -2.0], [2.0, -1.0]
            Number of points: 
                n = len(P) = 8
            Simplification Factor:
                1\ Example:
                    s_f = 1
                    P_new = [1.0, 1.0], [1.25, 2.0], [1.75, 2.0], [2.0, 1.0], [1.0, -1.0], [1.25, -2.0], [1.75, -2.0], [2.0, -1.0]
                    n = 8
                2\ Example:
                    simplification_factor = 2
                    P_new = [1.0, 1.0], [None], [1.75, 2.0], [None], [1.0, -1.0], [None], [1.75, -2.0], [2.0, -1.0] 
                    P_new = [1.0, 1.0], [1.75, 2.0], [1.0, -1.0], [1.75, -2.0], [2.0, -1.0]
                    n = 5

    Args:
        (1) P [Vector<float> nxm]: Input points to be simplified.
                                   Note:
                                    Where n is the number of dimensions of the point, and m is the number of points.
        (2) s_f [int]: The coefficient determines the simplification of the point vector.

    Returns:
        (1) parameter [Vector<float> nxn_s]: Simplified vector of points {P}.
                                             Note:
                                                Where n is the number of dimensions of the point, and n_s is number 
                                                of points after simplification.
    """

    P_aux = []
    P_aux.append(P[0])
    
    for i in range(1, P.shape[0] - 1):
        if i % s_f == 0:
            P_aux.append(P[i])

    if (P_aux[-1] == P[-1]).all() != True:
        P_aux.append(P[-1])

    return P_aux

def RDP_Simplification(P: tp.List[float], epsilon: float) -> tp.List[float]:
    """
    Description:
        A function to simplify a given array of points using the Ramer-Douglas-Peucker algorithm.

        Note:
            The Ramer-Douglas-Peucker (RDP) is an algorithm for reducing the number of points on a curve, that 
            is approximated by a series of points.
    
        The points must be in the following form:
            P = [p_0{x, y, ..}, p_1{x, y, ..}, ...]

    Args:
        (1) P [Vector<float> nxm]: Input points to be simplified.
                                   Note:
                                    Where n is the number of dimensions of the point, and m is the number of points.
        (2) epsilon [float]: The coefficient determines the similarity between the original a
                             and the approximated curve. 
                             
                             Note: 
                                epsilon > 0.0

    Returns:
        (1) parameter [Vector<float> nxn_a]: Simplified vector of points {P}.
                                             Note:
                                                Where n is the number of dimensions of the point, and n_a is number 
                                                of points after approximation.
    """

    try:
        assert epsilon > Mathematics.CONST_NULL

        d_max = 0.0; index = 0
        # Find the point with the maximum perpendicular distance.
        for i in range(1, P.shape[0]):
            d = Mathematics.Perpendicular_Distance(P[i], P[0], P[-1])
            if d > d_max:
                d_max = d; index = i

        """
        Description:
            If the maximum perpendicular distance is greater than epsilon, then recursively 
            simplify.
        """
        if d_max > epsilon:
            rdp_0 = RDP_Simplification(P[:(index + 1)], epsilon)
            rdp_1 = RDP_Simplification(P[index:], epsilon)

            return np.vstack((rdp_0[:-1], rdp_1))
        else:
            return np.vstack((P[0], P[-1]))

    except AssertionError as error:
        print(f'[ERROR] Information: {error}')
        print(f'[ERROR] The epsilon ({epsilon}) coefficient must be greater than zero.')

def __Uniformly_Spaced(P: tp.List[float]) -> tp.List[float]:
    """
    Description:
        A function to select parameters using the Uniformly-Spaced method.

        Reference:
            https://pages.mtu.edu/~shene/COURSES/cs3621/NOTES/INT-APP/PARA-uniform.html

    Args:
        (1) P [Vector<float> nxm]: Input points.
                                    Note:
                                        Where n is the number of dimensions of the point, and m is the number of points.

    Returns:
        (1) parameter [Vector<float> 1xn]: Selected parameters to be used to generate the vector of knots.
                                            Note:
                                                Where n is the number of dimensions of the point.
    """
        
    return np.linspace(0.0, 1.0, P.shape[0])

def __Chord_Length(P: tp.List[float]) -> tp.List[float]:
    """
    Description:
        A function to select parameters using the Chord-Length method.

        Reference:
            https://pages.mtu.edu/~shene/COURSES/cs3621/NOTES/INT-APP/PARA-chord-length.html

    Args:
        (1) P [Vector<float> nxm]: Input points.
                            Note:
                            Where n is the number of dimensions of the point, and m is the number of points.

    Returns:
        (1) parameter [Vector<float> 1xn]: Selected parameters to be used to generate the vector of knots.
                                            Note:
                                                Where n is the number of dimensions of the point.
    """

    # Express the number of control points.
    N = P.shape[0]
    
    t_k = np.zeros(N); L = 0.0
    for i in range(1, N):
        L_k = Mathematics.Euclidean_Norm(P[i] - P[i-1])
        t_k[i] = t_k[i-1] + L_k
        L += L_k

    return t_k / L

def __Centripetal(P: tp.List[float]) -> tp.List[float]:
    """
    Description:
        A function to select parameters using the Centripetal method.

        Reference:
            https://pages.mtu.edu/~shene/COURSES/cs3621/NOTES/INT-APP/PARA-centripetal.html
    
    Args:
        (1) P [Vector<float> nxm]: Input points.
                            Note:
                            Where n is the number of dimensions of the point, and m is the number of points.

    Returns:
        (1) parameter [Vector<float> 1xn]: Selected parameters to be used to generate the vector of knots.
                                            Note:
                                                Where n is the number of dimensions of the point.
    """
        
    # Express the number of control points.
    N = P.shape[0]
    
    t_k = np.zeros(N); L = 0.0
    for i in range(1, N):
        L_k = Mathematics.Euclidean_Norm(P[i] - P[i-1])
        t_k[i] = t_k[i-1] + L_k ** 0.5
        L += L_k ** 0.5

    return t_k / L

def __Knot_Parameter_Selection(P: tp.List[float], method: str) -> tp.List[float]:
    """
    Description:
        A function to select the parameters of the knot vector using an individual calculation method.

    Args:
        (1) P [Vector<float> nxm]: Input points.
                                   Note:
                                    Where n is the number of dimensions of the point, and m is the number of points.
        (2) method [string]: The method to be used to select the parameters. Possible string values can be: 
                             'Uniformly-Spaced', 'Chord-Length', 'Centripetal'.

    Returns:
        (1) parameter [Vector<float> 1xn]: Selected parameters to be used to generate the vector of knots.
                                            Note:
                                                Where n is the number of dimensions of the point.
    """
        
    return {
        'Uniformly-Spaced': lambda x: __Uniformly_Spaced(x),
        'Chord-Length': lambda x: __Chord_Length(x),
        'Centripetal': lambda x: __Centripetal(x)
    }[method](P)

def Generate_Knot_Vector(k: int, P: tp.List[float], method: str) -> tp.List[float]:
    """
    Description:
        A function to generate a normalized vector of knots from selected parameters using an individual 
        selection method (Uniformly-Spaced, Chord-Length, Centripetal).

        Reference:
            https://pages.mtu.edu/~shene/COURSES/cs3621/NOTES/INT-APP/PARA-knot-generation.html

    Args:
        (1) k [int]: Degree.
        (2) P [Vector<float> nxm]: Input points.
                                   Note:
                                    Where n is the number of dimensions of the point, and m is the number of points.
        (2) method [string]: The method to be used to select the parameters. Possible string values can be: 
                             'Uniformly-Spaced', 'Chord-Length', 'Centripetal'.

    Returns:
        (1) parameter [Vector<float> 1xn]: 
    """
        
    try:
        assert method in ['Uniformly-Spaced', 'Chord-Length', 'Centripetal']
        
        # Express the number of control points.
        N = P.shape[0]

        # Express the number of knots to be generated.
        t_n = N + k + 1
        
        # Select the parameters of the knot vector.
        t_param = __Knot_Parameter_Selection(P, method)

        # Generate a normalized vector of knots from the selected parameters.
        t = np.zeros(t_n)
        for i in range(t_n):
            if i < k + 1:
                t[i] = 0.0
            elif i >= N:
                t[i] = 1.0
            else:
                t[i] = np.sum(t_param[i-k:i]) / k

        return t

    except AssertionError as error:
        print(f'[ERROR] Information: {error}')
        print(f'[ERROR] Incorrect type of function input parameters. The generation method must correspond the name \
              given in the function description.')