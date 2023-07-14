# BPY (Blender as a python) [pip3 install bpy]
import bpy
# System (Default)
import sys
#   Add access if it is not in the system path.
if '../../' + 'src' not in sys.path:
    sys.path.append('../../' + 'src')
# Numpy (Array computing) [pip3 install numpy]
import numpy as np
# Custom Script:
#   ../Lib/Blender/Core
import Lib.Blender.Core
#   ../Lib/Blender/Utilities
import Lib.Blender.Utilities
#   ../Lib/Blender/Parameters/Camera
import Lib.Blender.Parameters.Camera
#   ../Lib/Interpolation/Bezier/Core
import Lib.Interpolation.Bezier.Core as Bezier
#   ../Lib/Interpolation/Utilities
import Lib.Interpolation.Utilities as Utilities
#   ../Lib/Transformation/Core
from Lib.Transformation.Core import Homogeneous_Transformation_Matrix_Cls as HTM_Cls, Euler_Angle_Cls as EA_Cls

"""
Description:
    Open Bezier_Curve.blend from the Blender folder and copy + paste this script and run it.

    Terminal:
        $ cd Documents/GitHub/Curve_Interpolation/Blender/Interpolation
        $ blender Bezier_Curve.blend
"""

"""
Description:
    Initialization of constants.
"""
# Set the structure of the main parameters of the camera.
CONST_CAMERA_TYPE = Lib.Blender.Parameters.Camera.Right_View_Camera_Parameters_Str
# Bezier curve interpolation parameters.
CONST_BEZIER_CURVE = {'method': 'Explicit', 'N': 100}
#   Visibility of the bounding box of the interpolated curve.
CONST_BOUNDING_BOX_VISIBILITY = False
#   Animation stop(t_0), start(t_1) time in seconds.
CONST_T_0 = 0.0
CONST_T_1 = 5.0

def main():
    """
    Description:
        A program for ...
    """

    # Deselect all objects in the current scene.
    Lib.Blender.Utilities.Deselect_All()

    # Remove animation data from objects (Clear keyframes).
    Lib.Blender.Utilities.Remove_Animation_Data()

    # Set the camera (object) transformation and projection.
    if Lib.Blender.Utilities.Object_Exist('Camera'):
        Lib.Blender.Utilities.Set_Camera_Properties('Camera', CONST_CAMERA_TYPE)

    """
    Description:
        Add the control point to the list, if it exists.

        Note:
            The position and number of control points is set by the user.
    """
    i = 0; P = []
    while True:
        P_name = f'Control_Point_{i}'
        if Lib.Blender.Utilities.Object_Exist(P_name) == True:
            P.append(np.array(bpy.data.objects[P_name].location))
        else:
            break     
        i += 1

    # Removes objects, if they exist.
    for _, obj_name in enumerate(['Bezier_Poly', 'Control_Points_Poly', 'Bounding_Box']):
        if Lib.Blender.Utilities.Object_Exist(obj_name) == True:
            Lib.Blender.Utilities.Remove_Object(obj_name)

    """
    Description:
        Visualization of 3-D (dimensional) linear interpolation of control points.
    """
    # Create a class to visualize a line segment.
    Control_Points_Poly = Lib.Blender.Core.Poly_3D_Cls('Control_Points_Poly', {'bevel_depth': 0.001, 'color': [0.05,0.05,0.05,1.0]}, 
                                                       {'visibility': False, 'radius': None, 'color': None})
    # Initialize the size (length) of the polyline data set.
    Control_Points_Poly.Initialization(i)
    # # Add coordinates to the polyline.
    for i, P_i in enumerate(P):           
        Control_Points_Poly.Add(i, P_i)
    # Visualization of a 3-D (dimensional) polyline in the scene.
    Control_Points_Poly.Visualization()

    # ...
    #   ...
    B_Cls = Bezier.Bezier_Cls(CONST_BEZIER_CURVE['method'], np.array(P), 
                              CONST_BEZIER_CURVE['N'])
    #   ...
    B = B_Cls.Interpolate()

    # Visibility of the bounding box of the interpolated curve.
    if CONST_BOUNDING_BOX_VISIBILITY == True:
        # ...
        B_Bounding_Box = B_Cls.Get_Bounding_Box_Parameters('Interpolated-Points')

        # Properties of the created object.
        bounding_box_properties = {'transformation': {'Size': 1.0, 
                                                      'Scale': [B_Bounding_Box['x_max'] - B_Bounding_Box['x_min'],
                                                                B_Bounding_Box['y_max'] - B_Bounding_Box['y_min'],
                                                                B_Bounding_Box['z_max'] - B_Bounding_Box['z_min']], 
                                                      'Location': [(B_Bounding_Box['x_max'] + B_Bounding_Box['x_min']) / 2.0,
                                                                   (B_Bounding_Box['y_max'] + B_Bounding_Box['y_min']) / 2.0,
                                                                   (B_Bounding_Box['z_max'] + B_Bounding_Box['z_min']) / 2.0]}, 
                                'material': {'RGBA': [1.0,0.25,0.0,1.0], 'alpha': 0.05}}
        # Create a primitive three-dimensional object (Cube -> Bounding-Box) with additional properties.
        Lib.Blender.Utilities.Create_Primitive('Cube', 'Bounding_Box', bounding_box_properties)

    # Obtain the position of the first/last control point.
    bpy.data.objects['Viewpoint_Control_Point_0'].location = B_Cls.P[0]
    bpy.data.objects['Viewpoint_Control_Point_n'].location = B_Cls.P[-1]
    # Obtain the orientation of the first/last object.
    #   Note:
    #       The orientation of the object is set by the user.
    q_0 = EA_Cls(np.array(bpy.data.objects['Viewpoint_Control_Point_0'].rotation_euler), 
                 'ZYX', np.float32).Get_Quaternion()
    q_1 = EA_Cls(np.array(bpy.data.objects['Viewpoint_Control_Point_n'].rotation_euler), 
                 'ZYX', np.float32).Get_Quaternion()

    """
    Description:
        Visualization of a 3-D (dimensional) interpolated curve (Bezier curve).
    """
    # Create a class to visualize a line segment.
    Bezier_Poly = Lib.Blender.Core.Poly_3D_Cls('Bezier_Poly', {'bevel_depth': 0.0015, 'color': [1.0,0.25,0.0,1.0]}, 
                                               {'visibility': True, 'radius': 0.004, 'color': [1.0,0.25,0.0,1.0]})
    # Initialize the size (length) of the polyline data set.
    Bezier_Poly.Initialization(N)
    # Add coordinates to the polyline.
    for i, B_i in enumerate(B):           
        Bezier_Poly.Add(i, B_i)
    # Visualization of a 3-D (dimensional) polyline in the scene.
    Bezier_Poly.Visualization()

    # The first frame on which the animation starts.
    np.int32(CONST_T_0 * (bpy.context.scene.render.fps / bpy.context.scene.render.fps_base))

    # Get the FPS (Frames Per Seconds) value from the Blender settings.
    fps = bpy.context.scene.render.fps / bpy.context.scene.render.fps_base

    """
    Description:
        Animation of a wiewpoint object on an interpolated curve.
    """
    for i, (B_i, t_i) in enumerate(zip(B, B_Cls.Time)):
        # Expression of the current animation frame.
        frame = np.int32((i/(B_Cls.N / (CONST_T_1 - CONST_T_0))) * fps)
        # Set scene frame.
        bpy.context.scene.frame_set(frame)

        # Obtain the spherical linear interpolation (Slerp) between the given quaternions.
        q = Utilities.Slerp('Quaternion', q_0, q_1, t_i)
        # Express the homogeneous transformation matrix of an object from position and rotation.
        T = HTM_Cls(None, np.float32).Rotation(q.all(), 'QUATERNION').Translation(B_i)

        # Set the transformation of the object (Viewpoint) to the current position/rotation of the curve.
        Lib.Blender.Utilities.Set_Object_Transformation('Viewpoint', T)

        # Insert a keyframe of the object (Viewpoint) into the frame at time t_{i}. 
        Lib.Blender.Utilities.Insert_Key_Frame('Viewpoint', 'matrix_basis', frame, 'ALL')

    # The last frame on which the animation stops.
    #   Note:
    #       Convert the time in seconds to the FPS value from the Blender settings.
    bpy.context.scene.frame_end = np.int32(CONST_T_1 * (bpy.context.scene.render.fps / bpy.context.scene.render.fps_base))

if __name__ == '__main__':
    main()