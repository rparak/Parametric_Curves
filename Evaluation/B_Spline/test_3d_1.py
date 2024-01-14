# System (Default)
import sys
#   Add access if it is not in the system path.
if '../../' + 'src' not in sys.path:
    sys.path.append('../../' + 'src')
# Numpy (Array computing) [pip3 install numpy]
import numpy as np
# OS (Operating system interfaces)
import os
# SciencePlots (Matplotlib styles for scientific plotting) [pip3 install SciencePlots]
import scienceplots
# Matplotlib (Visualization) [pip3 install matplotlib]
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.art3d
import matplotlib.patches as pat
# Custom Lib.:
#   ../Interpolation/B_Spline/Core
import Interpolation.B_Spline.Core as B_Spline
#   ../Primitives/Core
import Primitives.Core as Primitives

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
# Save the data to a file.
CONST_SAVE_DATA = False

def main():
    """
    Description:
        A program to visualize a parametric three-dimensional B-Spline curve of degree n.
    """

    # Locate the path to the project folder.
    project_folder = os.getcwd().split('Parametric_Curves')[0] + 'Parametric_Curves'

    # Input control points {P} in three-dimensional space.
    P = np.array([[1.00,  0.00, -1.00], 
                  [2.00, -0.75,  0.50], 
                  [3.00, -2.50,  1.00], 
                  [3.75, -1.25, -0.50], 
                  [4.00,  0.75,  1.50], 
                  [5.00,  1.00, -1.50]], dtype=np.float64)

    # Initialization of a specific class to work with B-Spline curves.
    S_Cls = B_Spline.B_Spline_Cls(CONST_B_SPLINE['n'], CONST_B_SPLINE['method'], P, 
                                  CONST_B_SPLINE['N'])
    
    # Interpolation of parametric B-Spline curve.
    S = S_Cls.Interpolate()
    
    # Obtain the arc length L(x) of the general parametric curve.
    L = S_Cls.Get_Arc_Length()

    # Set the parameters for the scientific style.
    plt.style.use(['science'])

    # Create a figure.
    figure = plt.figure()
    ax = figure.add_subplot(projection='3d')

    # Visualization of relevant structures.
    ax.plot(S_Cls.P[:, 0], S_Cls.P[:, 1], S_Cls.P[:, 2], 'o--', color='#d0d0d0', linewidth=1.0, markersize = 8.0, 
            markeredgewidth = 4.0, markerfacecolor = '#ffffff', label='Control Points')
    ax.plot(S[:, 0], S[:, 1], S[:, 2], '.-', color='#ffbf80', linewidth=1.5, markersize = 8.0, 
            markeredgewidth = 2.0, markerfacecolor = '#ffffff', label=f'B-Spline (n = {S_Cls.n}, N = {S_Cls.N}, L = {L:.03})')

    # Visibility of the bounding box of the interpolated curve.
    if CONST_BOUNDING_BOX['visibility'] == True:
        # Get the bounding parameters (min, max) selected by the user.
        Bounding_Box = S_Cls.Get_Bounding_Box_Parameters(CONST_BOUNDING_BOX['limitation'])

        # Create a primitive three-dimensional object (Cube -> Bounding-Box) with additional properties.
        Box_Cls = Primitives.Box_Cls([0.0, 0.0, 0.0], [Bounding_Box['x_max'] - Bounding_Box['x_min'], 
                                                       Bounding_Box['y_max'] - Bounding_Box['y_min'], 
                                                       Bounding_Box['z_max'] - Bounding_Box['z_min']])
        
        # Change the position of the bounding box.
        Bounding_Box_new = np.zeros(Box_Cls.Faces.shape)
        for i, Box_Faces_i in enumerate(Box_Cls.Faces):
            for j, Box_Faces_ij in enumerate(Box_Faces_i):
                Bounding_Box_new[i, j, :] = Box_Faces_ij + [(Bounding_Box['x_max'] + Bounding_Box['x_min']) / 2.0, 
                                                            (Bounding_Box['y_max'] + Bounding_Box['y_min']) / 2.0, 
                                                            (Bounding_Box['z_max'] + Bounding_Box['z_min']) / 2.0]
                
        # Add a 3D collection object to the plot.
        edgcolor = '#ebebeb' if CONST_BOUNDING_BOX['limitation'] == 'Control-Points' else '#ffd8b2'
        ax.add_collection3d(mpl_toolkits.mplot3d.art3d.Poly3DCollection(Bounding_Box_new, linewidths=1.5, edgecolors=edgcolor, 
                                                                        facecolors = '#ffffff', alpha=0.01))
        
    # Set parameters of the graph (plot).
    ax.set_title(f'B-Spline Interpolation in {P.shape[1]}-Dimensional Space', fontsize=25, pad=25.0)
    #   Set the x ticks.
    ax.set_xticks(np.arange(np.min(P[:, 0]) - 0.5, np.max(P[:, 0]) + 1.0, 0.5))
    #   Set the y ticks.
    ax.set_yticks(np.arange(np.min(P[:, 1]) - 0.5, np.max(P[:, 1]) + 1.0, 0.5))
    #   Set the z ticks.
    ax.set_zticks(np.arange(np.min(P[:, 2]) - 0.5, np.max(P[:, 2]) + 1.0, 0.5))
    #   Limits.
    ax.set_xlim(np.minimum.reduce(S_Cls.P[:, 0]) - 0.5, np.maximum.reduce(S_Cls.P[:, 0]) + 1.0)
    ax.xaxis.pane.set_color((1.0, 1.0, 1.0, 1.0))
    ax.set_ylim(np.minimum.reduce(S_Cls.P[:, 1]) - 0.5, np.maximum.reduce(S_Cls.P[:, 1]) + 1.0)
    ax.yaxis.pane.set_color((1.0, 1.0, 1.0, 1.0))
    ax.set_zlim(np.minimum.reduce(S_Cls.P[:, 2]) - 0.5, np.maximum.reduce(S_Cls.P[:, 2]) + 1.0)
    ax.zaxis.pane.set_color((1.0, 1.0, 1.0, 1.0))
    #   Label.
    ax.set_xlabel(r'x-axis in meters', fontsize=15, labelpad=10); ax.set_ylabel(r'y-axis in meters', fontsize=15, labelpad=10) 
    ax.set_zlabel(r'z-axis in meters', fontsize=15, labelpad=10) 
    #   Set parameters of the visualization.
    ax.xaxis._axinfo['grid'].update({'linewidth': 0.15, 'linestyle': '--'})
    ax.yaxis._axinfo['grid'].update({'linewidth': 0.15, 'linestyle': '--'})
    ax.zaxis._axinfo['grid'].update({'linewidth': 0.15, 'linestyle': '--'})
    #   Set the Axes box aspect.
    ax.set_box_aspect(None, zoom=0.90)
    # Get handles and labels for the legend.
    handles, labels = plt.gca().get_legend_handles_labels()
    if CONST_BOUNDING_BOX['visibility'] == True:
        # Add a bounding box legend.
        handles.append(pat.Rectangle(xy = (0.0, 0.02), width = 0.0, height = 0.0, facecolor = 'none',
                                    edgecolor = edgcolor, linewidth = 1.5))
        labels.append('Bounding Box')
    # Remove duplicate labels.
    legend = dict(zip(labels, handles))
    # Show the labels (legends) of the graph.
    ax.legend(legend.values(), legend.keys(), fontsize=10.0)

    if CONST_SAVE_DATA == True:
        # Set the full scree mode.
        plt.get_current_fig_manager().full_screen_toggle()

        # Save the results.
        plt.savefig(f'{project_folder}/images/B_Spline/test_3d_1_0.png', format='png', dpi=300)
    else:
        # Show the result.
        plt.show()

if __name__ == "__main__":
    sys.exit(main())
