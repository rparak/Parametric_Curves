# Numpy (Array computing) [pip3 install numpy]
import numpy as np
# Typing (Support for type hints)
import typing as tp
# Custom Script:
#   ../Lib/Transformation/Core
from Lib.Transformation.Core import Homogeneous_Transformation_Matrix_Cls as HTM_Cls

"""
Description:
    Initialization of constants.
"""
# The calculation will be performed in three-dimensional (3D) space.
CONST_DIMENSION = 3
# Shape (Box):
#   Vertices: 8; Space: 3D;
CONST_BOX_SHAPE = (8, CONST_DIMENSION)

class Box_Cls(object):
    """
    Description:
        A specific class for working with box (or cuboid) as a primitive object.

        Note:
            The box (or cuboid) is the 3D version of a rectangle. We can define a three-dimensional box by the origin 
            and a size.

    Initialization of the Class:
        Args:
            (1) origin [Vector<float> 1x3]: The origin of the object.
            (2) size [Vector<float> 1x3]: The size of the object.

        Example:
            Initialization:
                # Assignment of the variables.
                origin = [0.0, 0.0, 0.0]
                size   = [1.0, 1.0, 1.0]

                # Initialization of the class.
                Cls = Box_Cls(origin, size)

            Features:
                # Properties of the class.
                Cls.Size; Cls.T; Cls.Vertices
    """
        
    def __init__(self, origin: tp.List[float] = [0.0] * CONST_DIMENSION, size: tp.List[float] = [0.0] * CONST_DIMENSION) -> None:
        # << PRIVATE >> #
        self.__size = np.array(size, np.float32)

        # Calculate the object's centroid from the object's origin.
        self.__centroid = np.array([0.0] * CONST_DIMENSION, np.float32) - np.array(origin, np.float32)

        # Convert the initial object sizes to a transformation matrix.
        self.__T_Size = HTM_Cls(None, np.float32).Scale(self.__size)

        # Calculate the vertices of the box defined by the input parameters of the class.
        self.__vertices = np.zeros(CONST_BOX_SHAPE, dtype=np.float32)
        for i, verts_i in enumerate(self.__Get_Init_Vertices()):
            self.__vertices[i, :] = (self.__T_Size.all() @ np.append(verts_i - origin, 1.0).tolist())[0:3]
            
    @staticmethod
    def __Get_Init_Vertices() -> tp.List[tp.List[float]]:
        """
        Description:
            Helper function to get the initial vertices of the object.

            Note: 
                Lower Base: A {id: 0}, B {id: 1}, C {id: 2}, D {id: 3}
                Upper Base: E {id: 4}, F {id: 5}, G {id: 6}, H {id: 7}

        Returns:
            (1) parameter [Vector<float> 8x3]: Vertices of the object.
        """
 
        return np.array([[0.5, -0.5, -0.5], [0.5, 0.5, -0.5], [-0.5, 0.5, -0.5], [-0.5, -0.5, -0.5],
                         [0.5, -0.5,  0.5], [0.5, 0.5,  0.5], [-0.5, 0.5,  0.5], [-0.5, -0.5,  0.5]], dtype=np.float32)

    @property
    def Size(self) -> tp.List[float]:
        """
        Description:
            Get the size of the box in the defined space.

        Returns:
            (1) parameter [Vector<float> 1x3]: Box size (X, Y, Z).
        """
                
        return self.__size

    @property
    def Vertices(self) -> tp.List[tp.List[float]]:
        """
        Description:
            Get the vertices of the object.

        Returns:
            (1) parameter [Vector<float> 8x3]: Vertices of the object.
        """

        return self.__vertices
    
    @property
    def Faces(self) -> tp.List[tp.List[tp.List[float]]]:
        """
        Description:
            Get the faces of the object.

        Returns:
            (1) parameter [Vector<float> 6x4x3]: Faces of the object.
        """

        return np.array([[self.__vertices[0], self.__vertices[1], self.__vertices[2], self.__vertices[3]],
                         [self.__vertices[4], self.__vertices[5], self.__vertices[6], self.__vertices[7]],
                         [self.__vertices[3], self.__vertices[0], self.__vertices[4], self.__vertices[7]],
                         [self.__vertices[2], self.__vertices[1], self.__vertices[5], self.__vertices[6]],
                         [self.__vertices[0], self.__vertices[1], self.__vertices[5], self.__vertices[4]],
                         [self.__vertices[3], self.__vertices[2], self.__vertices[6], self.__vertices[7]]], dtype=np.float32)

    @property
    def T(self) -> HTM_Cls:
        """
        Description:
            Get the object's transformation with zero rotation.

        Returns:
            (1) parameter [HTM_Cls(object) -> Matrix<float> 4x4]: Homogeneous transformation matrix 
                                                                  of the object.
        """

        return HTM_Cls(None, np.float32).Translation(self.__centroid)