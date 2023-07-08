import numpy as np
# System (Default)
import sys
#   Add access if it is not in the system path.
if '../' + 'src' not in sys.path:
    sys.path.append('../..')
import Lib.Interpolation.Utilities as Utilities

p = np.array([[ 1., 0.], [ 1., 1.], [ 0., 2.], [ -0.5, 3.], [ 1., 1.], [ 3., 1.]])
k = 3

knot_10 = Utilities.Generate_Knot_Vector(k, p, 'Chord-Length')
print(knot_10)
