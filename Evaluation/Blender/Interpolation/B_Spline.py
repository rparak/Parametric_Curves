# BPY (Blender as a python) [pip3 install bpy]
import bpy
# System (Default)
import sys
#   Add access if it is not in the system path.
if '../../' + 'src' not in sys.path:
    sys.path.append('../../' + 'src')
# Numpy (Array computing) [pip3 install numpy]
import numpy as np
# Custom Lib.:
#   ../Blender/Core
import Blender.Core
#   ../Blender/Utilities
import Blender.Utilities
#   ../Blender/Parameters/Camera
import Blender.Parameters.Camera
#   ../Interpolation/B_Spline/Core
import Interpolation.B_Spline.Core as B_Spline
#   ../Interpolation/Utilities
import Interpolation.Utilities as Utilities
#   ../Transformation/Core
from Transformation.Core import Homogeneous_Transformation_Matrix_Cls as HTM_Cls, Euler_Angle_Cls as EA_Cls

"""
Description:
    Open B_Spline.blend from the Blender folder and copy + paste this script and run it.

    Terminal:
        $ cd Documents/GitHub/Parametric_Curves/Blender/Interpolation
        $ blender B_Spline.blend
"""

"""
Description:
    Initialization of constants.
"""
# Set the structure of the main parameters of the camera.
CONST_CAMERA_TYPE = Blender.Parameters.Camera.Right_View_Camera_Parameters_Str
# B-Spline interpolation parameters.
CONST_B_SPLINE = {'n': 3, 'N': 50, 'method': 'Chord-Length'}
# Visibility of the bounding box:
#   'limitation': 'Control-Points' or 'Interpolated-Points'
CONST_BOUNDING_BOX = {'visibility': False, 'limitation': 'Control-Points'}
#   Animation stop(x_0), start(x_1) time in seconds.
CONST_T_0 = 0.0
CONST_T_1 = 5.0

def main():
    """
    Description:
        A program to visualize a parametric three-dimensional B-Spline curve of degree n.

        Note:
            (1) The position and number of control points is set by the user.
                Object Name: 'Control_Point_{i}', where i is the identification number of control point.
            (2) The orientation of the object is set by the user.
                Object Name: 'Viewpoint_Control_Point_0', 'Viewpoint_Control_Point_n', where 0 denotes the first point 
                              and n the last point.
    """

    # Deselect all objects in the current scene.
    Blender.Utilities.Deselect_All()

    # Remove animation data from objects (Clear keyframes).
    Blender.Utilities.Remove_Animation_Data()

    # Set the camera (object) transformation and projection.
    if Blender.Utilities.Object_Exist('Camera'):
        Blender.Utilities.Set_Camera_Properties('Camera', CONST_CAMERA_TYPE)

    """
    Description:
        Add the control point to the list, if it exists.

        Note:
            The position and number of control points is set by the user.
    """
    i = 0; P = []
    while True:
        P_name = f'Control_Point_{i}'
        if Blender.Utilities.Object_Exist(P_name) == True:
            P.append(np.array(bpy.data.objects[P_name].location))
        else:
            break     
        i += 1

    # Initialization of a specific class to work with B-Spline curves.
    S_Cls = B_Spline.B_Spline_Cls(CONST_B_SPLINE['n'], CONST_B_SPLINE['method'], np.array(P),
                                  CONST_B_SPLINE['N'])
    
    # Interpolation of parametric B-Spline curve.
    S = S_Cls.Interpolate()

    # Removes objects, if they exist.
    for _, obj_name in enumerate(['B-Spline_Poly', 'Control_Points_Poly', 'Bounding_Box']):
        if Blender.Utilities.Object_Exist(obj_name) == True:
            Blender.Utilities.Remove_Object(obj_name)

    """
    Description:
        Visualization of 3-D (dimensional) linear interpolation of control points.
    """
    # Create a class to visualize a line segment.
    Control_Points_Poly = Blender.Core.Poly_3D_Cls('Control_Points_Poly', {'bevel_depth': 0.001, 'color': [0.05,0.05,0.05,1.0]}, 
                                                       {'visibility': False, 'radius': None, 'color': None})
    # Initialize the size (length) of the polyline data set.
    Control_Points_Poly.Initialization(i)
    # # Add coordinates to the polyline.
    for i, P_i in enumerate(S_Cls.P):           
        Control_Points_Poly.Add(i, P_i)
    # Visualization of a 3-D (dimensional) polyline in the scene.
    Control_Points_Poly.Visualization()

    # Visibility of the bounding box of the interpolated curve.
    if CONST_BOUNDING_BOX['visibility'] == True:
        # Get the bounding parameters (min, max) selected by the user.
        Bounding_Box = S_Cls.Get_Bounding_Box_Parameters(CONST_BOUNDING_BOX['limitation'])

        # Set the color of the bounding box.
        color = [0.05,0.05,0.05,1.0] if CONST_BOUNDING_BOX['limitation'] == 'Control-Points' else [1.0,0.25,0.0,1.0]

        # Properties of the created object.
        bounding_box_properties = {'transformation': {'Size': 1.0, 
                                                      'Scale': [Bounding_Box['x_max'] - Bounding_Box['x_min'],
                                                                Bounding_Box['y_max'] - Bounding_Box['y_min'],
                                                                Bounding_Box['z_max'] - Bounding_Box['z_min']], 
                                                      'Location': [(Bounding_Box['x_max'] + Bounding_Box['x_min']) / 2.0,
                                                                   (Bounding_Box['y_max'] + Bounding_Box['y_min']) / 2.0,
                                                                   (Bounding_Box['z_max'] + Bounding_Box['z_min']) / 2.0]}, 
                                'material': {'RGBA': color, 'alpha': 0.05}}
        # Create a primitive three-dimensional object (Cube -> Bounding-Box) with additional properties.
        Blender.Utilities.Create_Primitive('Cube', 'Bounding_Box', bounding_box_properties)

    # Obtain the position of the first/last control point.
    bpy.data.objects['Viewpoint_Control_Point_0'].location = S_Cls.P[0]
    bpy.data.objects['Viewpoint_Control_Point_n'].location = S_Cls.P[-1]
    # Obtain the orientation of the first/last object.
    #   Note:
    #       The orientation of the object is set by the user.
    q_0 = EA_Cls(np.array(bpy.data.objects['Viewpoint_Control_Point_0'].rotation_euler), 
                 'ZYX', np.float64).Get_Quaternion()
    q_1 = EA_Cls(np.array(bpy.data.objects['Viewpoint_Control_Point_n'].rotation_euler), 
                 'ZYX', np.float64).Get_Quaternion()
                
    """
    Description:
        Visualization of a 3-D (dimensional) interpolated curve (B-Spline).
    """
    # Create a class to visualize a line segment.
    B_Spline_Poly = Blender.Core.Poly_3D_Cls('B-Spline_Poly', {'bevel_depth': 0.0015, 'color': [1.0,0.25,0.0,1.0]}, 
                                                 {'visibility': True, 'radius': 0.004, 'color': [1.0,0.25,0.0,1.0]})
    # Initialize the size (length) of the polyline data set.
    B_Spline_Poly.Initialization(S_Cls.N)
    # Add coordinates to the polyline.
    for i, S_i in enumerate(S):           
        B_Spline_Poly.Add(i, S_i)
    # Visualization of a 3-D (dimensional) polyline in the scene.
    B_Spline_Poly.Visualization()

    # Get the FPS (Frames Per Seconds) value from the Blender settings.
    fps = bpy.context.scene.render.fps / bpy.context.scene.render.fps_base

    # The first frame on which the animation starts.
    bpy.context.scene.frame_start = np.int32(CONST_T_0 * fps)

    """
    Description:
        Animation of a wiewpoint object on an interpolated curve.
    """
    for i, (S_i, t_i) in enumerate(zip(S, S_Cls.x)):
        # Expression of the current animation frame.
        frame = np.int32((i/(S_Cls.N / (CONST_T_1 - CONST_T_0))) * fps)
        # Set scene frame.
        bpy.context.scene.frame_set(frame)

        # Obtain the spherical linear interpolation (Slerp) between the given quaternions.
        q = Utilities.Slerp('Quaternion', q_0, q_1, t_i)
        # Express the homogeneous transformation matrix of an object from position and rotation.
        T = HTM_Cls(None, np.float64).Rotation(q.all(), 'QUATERNION').Translation(S_i)

        # Set the transformation of the object (Viewpoint) to the current position/rotation of the curve.
        Blender.Utilities.Set_Object_Transformation('Viewpoint', T)

        # Insert a keyframe of the object (Viewpoint) into the frame at time x_{i}. 
        Blender.Utilities.Insert_Key_Frame('Viewpoint', 'matrix_basis', frame, 'ALL')

    # The last frame on which the animation stops.
    bpy.context.scene.frame_end = np.int32(CONST_T_1 * fps)

if __name__ == '__main__':
    main()
