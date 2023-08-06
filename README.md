# An Open-Source Parametric Curves Library Useful for Robotics Applications

<p align="center">
<img src=https://github.com/rparak/Parametric_Curves/blob/main/images/Parametric_Curves_Background.png width="800" height="350">
</p>

## Requirements

**Programming Language**

```bash
Python
```

**Import Libraries**
```bash
More information can be found in the individual scripts (.py).
```

**Supported on the following operating systems**
```bash
Windows, Linux, macOS
```

## Project Description
An open-source library of parametric curves interpolation (Bézier, B-Spline) useful for robotics applications, such as path planning, etc. The library provides access to specific classes for working with two types of parametric curves: Bézier curves and B-Spline curves. The specific classes focus on the problem of interpolating both two-dimensional and three-dimensional curves from input control points.

The classes also include the methods to calculate the derivative of the individual curves, arc-length, bounding box, simplification of the curves, and much more. The B-Spline interpolation class contains a function to optimize control points using the least squares method.

```bash
Path (Bézier): ..\Collision_Detection\src\Lib\Interpolation\Bezier\Core.py
Path (B-Spline): ..\Collision_Detection\src\Lib\Interpolation\B_Spline\Core.py
```

In particular, the library focuses on solving the path planning problem of the industrial/collaborative robotic arms. But, as an open-source library, it can be used for other tasks, as creativity knows no limits.

The repository also contains a transformation library with the necessary project-related functions. See link below.

[/rparak/Transformation](https://github.com/rparak/Transformation)

The library can be used within the Robot Operating System (ROS), Blender, PyBullet, Nvidia Isaac, or any program that allows Python as a programming language.

## Bézier Curves

**Bernstein Polynomials**

```bash
$ /> cd Documents/GitHub/Parametric_Curves/src/Evaluation/Bezier
$ ../Collision_Detection/Blender> python3 Bernstein_Polynomials.py
```

<p align="center">
    <img src=https://github.com/rparak/Parametric_Curves/blob/main/images/Bezier/Bernstein_Polynomials.png width="600" height="350">
</p>


**Demonstration of two-dimensional (2D) Curves**

```bash
$ /> cd Documents/GitHub/Parametric_Curves/src/Evaluation/Bezier
$ ../Collision_Detection/Blender> python3 test_2d_1.py
```

<p align="center">
    <img src=https://github.com/rparak/Parametric_Curves/blob/main/images/Bezier/test_2d_1_0.png width="600" height="350">
    <img src=https://github.com/rparak/Parametric_Curves/blob/main/images/Bezier/test_2d_1_1.png width="600" height="350">
</p>

**Demonstration of three-dimensional (3D) Curves**

```bash
$ /> cd Documents/GitHub/Parametric_Curves/src/Evaluation/Bezier
$ ../Collision_Detection/Blender> python3 test_3d_1.py
```

<p align="center">
    <img src=https://github.com/rparak/Parametric_Curves/blob/main/images/Bezier/test_3d_1_0.png width="600" height="600">
    <img src=https://github.com/rparak/Parametric_Curves/blob/main/images/Bezier/test_3d_1_1.png width="600" height="600">
</p>

A simple program that describes how to work with the library can be found below. The whole program is located in the individual evaluation folder.

```py 
# System (Default)
import sys
# Numpy (Array computing) [pip3 install numpy]
import numpy as np
# Matplotlib (Visualization) [pip3 install matplotlib]
import matplotlib.pyplot as plt
# Custom Script:
#   ../Lib/Interpolation/Bezier/Core
import Lib.Interpolation.Bezier.Core as Bezier
"""
Description:
    Initialization of constants.
"""
# Bezier curve interpolation parameters.
#   'method': The name of the method to be used to interpolate the parametric curve.
#               method = 'Explicit' or 'Polynomial'.
#   N: The number of points to be generated in the interpolation function.
CONST_BEZIER_CURVE = {'method': 'Explicit', 'N': 100}
# Visibility of the bounding box:
#   'limitation': 'Control-Points' or 'Interpolated-Points'
CONST_BOUNDING_BOX = {'visibility': False, 'limitation': 'Control-Points'}

def main():
    """
    Description:
        A program to visualize a parametric two-dimensional Bézier curve of degree n.
    """

    # Input control points {P} in two-dimensional space.
    P = np.array([[1.00,  0.00], 
                  [2.00, -0.75], 
                  [3.00, -2.50], 
                  [3.75, -1.25], 
                  [4.00,  0.75], 
                  [5.00,  1.00]], dtype=np.float32)

    # Initialization of a specific class to work with Bézier curves.
    B_Cls = Bezier.Bezier_Cls(CONST_BEZIER_CURVE['method'], P, 
                              CONST_BEZIER_CURVE['N'])
    
    # Interpolation of parametric Bézier curve.
    B = B_Cls.Interpolate()

    # Obtain the arc length L(x) of the general parametric curve.
    L = B_Cls.Get_Arc_Length()

    # Create a figure.
    _, ax = plt.subplots()

    # Visualization of relevant structures.
    ax.plot(P[:, 0], P[:, 1], 'o--', color='#d0d0d0', linewidth=1.0, markersize = 8.0, 
            markeredgewidth = 4.0, markerfacecolor = '#ffffff', label='Control Points')
    ax.plot(B[:, 0], B[:, 1], '.-', color='#ffbf80', linewidth=1.5, markersize = 8.0, 
            markeredgewidth = 2.0, markerfacecolor = '#ffffff', label=f'Bézier Curve (N = {B_Cls.N}, L = {L:.03})')

    # Show the result.
    plt.show()

if __name__ == '__main__':
    sys.exit(main())
```

## B-Spline Curves

**Basic Functions**

```bash
$ /> cd Documents/GitHub/Parametric_Curves/src/Evaluation/B_Spline
$ ../Collision_Detection/Blender> python3 Basic_Functions.py
```

<p align="center">
    <img src=https://github.com/rparak/Parametric_Curves/blob/main/images/B_Spline/Basic_Functions.png width="600" height="350">
</p>


**Demonstration of two-dimensional (2D) Curves**

```bash
$ /> cd Documents/GitHub/Parametric_Curves/src/Evaluation/B_Spline
$ ../Collision_Detection/Blender> python3 test_2d_1.py
```

<p align="center">
    <img src=https://github.com/rparak/Parametric_Curves/blob/main/images/B_Spline/test_2d_1_0.png width="600" height="350">
    <img src=https://github.com/rparak/Parametric_Curves/blob/main/images/B_Spline/test_2d_1_1.png width="600" height="350">
</p>

```bash
$ /> cd Documents/GitHub/Parametric_Curves/src/Evaluation/B_Spline
$ ../Collision_Detection/Blender> python3 test_2d_2.py
```

<p align="center">
    <img src=https://github.com/rparak/Parametric_Curves/blob/main/images/B_Spline/test_2d_2_0.png width="600" height="350">
</p>

**Demonstration of three-dimensional (3D) Curves**

```bash
$ /> cd Documents/GitHub/Parametric_Curves/src/Evaluation/B_Spline
$ ../Collision_Detection/Blender> python3 test_3d_1.py
```

<p align="center">
    <img src=https://github.com/rparak/Parametric_Curves/blob/main/images/B_Spline/test_3d_1_0.png width="600" height="600">
    <img src=https://github.com/rparak/Parametric_Curves/blob/main/images/B_Spline/test_3d_1_1.png width="600" height="600">
</p>

```bash
$ /> cd Documents/GitHub/Parametric_Curves/src/Evaluation/B_Spline
$ ../Collision_Detection/Blender> python3 test_3d_2.py
```

<p align="center">
    <img src=https://github.com/rparak/Parametric_Curves/blob/main/images/B_Spline/test_3d_2_0.png width="600" height="600">
</p>

A simple program that describes how to work with the library can be found below. The whole program is located in the individual evaluation folder.

```py 
# System (Default)
import sys
# Numpy (Array computing) [pip3 install numpy]
import numpy as np
# Matplotlib (Visualization) [pip3 install matplotlib]
import matplotlib.pyplot as plt
# Custom Script:
#   ../Lib/Interpolation/B_Spline/Core
import Lib.Interpolation.B_Spline.Core as B_Spline

"""
Description:
    Initialization of constants.
"""
# B-Spline interpolation parameters.
#   n: Degree of a polynomial.
#   N: The number of points to be generated in the interpolation function.
#   'method': The method to be used to select the parameters of the knot vector. 
#               method = 'Uniformly-Spaced', 'Chord-Length' or 'Centripetal'.
CONST_B_SPLINE = {'n': 3, 'N': 100, 'method': 'Chord-Length'}
# Visibility of the bounding box:
#   'limitation': 'Control-Points' or 'Interpolated-Points'
CONST_BOUNDING_BOX = {'visibility': False, 'limitation': 'Control-Points'}

def main():
    """
    Description:
        A program to visualize a parametric two-dimensional B-Spline curve of degree n.
    """

    # Input control points {P} in two-dimensional space.
    P = np.array([[1.00,  0.00], 
                  [2.00, -0.75], 
                  [3.00, -2.50], 
                  [3.75, -1.25], 
                  [4.00,  0.75], 
                  [5.00,  1.00]], dtype=np.float32)

    # Initialization of a specific class to work with B-Spline curves.
    S_Cls = B_Spline.B_Spline_Cls(CONST_B_SPLINE['n'], CONST_B_SPLINE['method'], P, 
                                  CONST_B_SPLINE['N'])
    
    # Interpolation of parametric B-Spline curve.
    S = S_Cls.Interpolate()

    # Obtain the arc length L(x) of the general parametric curve.
    L = S_Cls.Get_Arc_Length()

    # Create a figure.
    _, ax = plt.subplots()

    # Visualization of relevant structures.
    ax.plot(P[:, 0], P[:, 1], 'o--', color='#d0d0d0', linewidth=1.0, markersize = 8.0, 
            markeredgewidth = 4.0, markerfacecolor = '#ffffff', label='Control Points')
    ax.plot(S[:, 0], S[:, 1], '.-', color='#ffbf80', linewidth=1.5, markersize = 8.0, 
            markeredgewidth = 2.0, markerfacecolor = '#ffffff', label=f'B-Spline (n = {S_Cls.n}, N = {S_Cls.N}, L = {L:.03})')

    # Show the result.
    plt.show()

if __name__ == '__main__':
    sys.exit(main())
```

## Blender
The library for parametric curves can also be used within the Blender software. See the instructions below for more information.

**Bézier Curves**

A description of how to run a program to visualize a parametric three-dimensional Bézier curve of degree n.
1. Open Bezier_Curve.blend from the Blender folder.
2. Copy and paste the script from the evaluation folder (../Bezier_Curve.py).
3. Run it and evaluate the results.
   
```bash
$ /> cd Documents/GitHub/Parametric_Curves/Blender/Interpolation
$ ../Collision_Detection/Blender> blender Bezier_Curve.blend
```

<p align="center">
<img src=https://github.com/rparak/Parametric_Curves/blob/main/images/Blender/Bezier_Curve.png width="650" height="350">
</p>

**B-Spline Curves**

A description of how to run a program to visualize a parametric three-dimensional B-Spline curve of degree n.
1. Open B_Spline.blend from the Blender folder.
2. Copy and paste the script from the evaluation folder (../B_Spline.py).
3. Run it and evaluate the results.
   
```bash
$ /> cd Documents/GitHub/Parametric_Curves/Blender/Interpolation
$ ../Collision_Detection/Blender> blender B_Spline.blend
```

<p align="center">
<img src=https://github.com/rparak/Parametric_Curves/blob/main/images/Blender/B_Spline.png width="650" height="350">
</p>

## YouTube

[<p align="center"><img src=https://github.com/rparak/Parametric_Curves/blob/main/images/Blender/B_Spline.png width="650" height="350"></p>](https://www.youtube.com/watch?v=hkEybI5IzpE&t=61s)

## Contact Info
Roman.Parak@outlook.com

## Citation (BibTex)
```bash
@misc{RomanParak_DataConverter,
  author = {Roman Parak},
  title = {An open-source parametric curves library useful for robotics applications},
  year = {2023},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://https://github.com/rparak/Parametric_Curves}}
}
```

## License
[MIT](https://choosealicense.com/licenses/mit/)
