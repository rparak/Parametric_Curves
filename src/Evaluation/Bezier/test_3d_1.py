# System (Default)
import sys
#   Add access if it is not in the system path.
if '../' + 'src' not in sys.path:
    sys.path.append('../..')
# Numpy (Array computing) [pip3 install numpy]
import numpy as np
# SciencePlots (Matplotlib styles for scientific plotting) [pip3 install SciencePlots]
import scienceplots
# Matplotlib (Visualization) [pip3 install matplotlib]
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.art3d
import matplotlib.patches as pat
# Custom Script:
#   ../Lib/Interpolation/Bezier/Core
import Lib.Interpolation.Bezier.Core as Bezier
#   ../Lib/Utilities/Primitives
import Lib.Utilities.Primitives as Primitives

def main():
    # ...
    #   ...
    method = 'Explicit'; N = 100
    #   ...
    P = np.array([[1.00,  0.00, -1.00], 
                  [2.00, -0.75,  0.50], 
                  [3.00, -2.50,  1.00], 
                  [3.75, -1.25, -0.50], 
                  [4.00,  0.75,  1.50], 
                  [5.00,  1.00, -1.50]])

    # ...
    B_Cls = Bezier.Bezier_Cls(method, P, N)
    # ...
    B = B_Cls.Interpolate()
    # ...
    L = B_Cls.Get_Arc_Length()
    # ...
    B_Bounding_Box = B_Cls.Get_Bounding_Box_Parameters('Interpolated-Points')
    
    # ...
    Box_Cls = Primitives.Box_Cls([0.0, 0.0, 0.0], [B_Bounding_Box['x_max'] - B_Bounding_Box['x_min'], 
                                                   B_Bounding_Box['y_max'] - B_Bounding_Box['y_min'], 
                                                   B_Bounding_Box['z_max'] - B_Bounding_Box['z_min']])
    #   ...
    Bounding_Box = np.zeros(Box_Cls.Faces.shape)
    for i, Box_Faces_i in enumerate(Box_Cls.Faces):
        for j, Box_Faces_ij in enumerate(Box_Faces_i):
            Bounding_Box[i, j, :] = Box_Faces_ij + [(B_Bounding_Box['x_max'] + B_Bounding_Box['x_min']) / 2.0, 
                                                    (B_Bounding_Box['y_max'] + B_Bounding_Box['y_min']) / 2.0, 
                                                    (B_Bounding_Box['z_max'] + B_Bounding_Box['z_min']) / 2.0]

    # Set the parameters for the scientific style.
    plt.style.use(['science'])

    # Create a figure.
    figure = plt.figure()
    ax = figure.add_subplot(projection='3d')

    # ...
    ax.plot(B_Cls.P[:, 0], B_Cls.P[:, 1], B_Cls.P[:, 2], 'o--', color='#d0d0d0', linewidth=1.0, markersize = 8.0, 
            markeredgewidth = 4.0, markerfacecolor = '#ffffff', label='Control Points')
    # ...
    ax.plot(B[:, 0], B[:, 1], B[:, 2], '.-', color='#ffbf80', linewidth=1.5, markersize = 8.0, 
            markeredgewidth = 2.0, markerfacecolor = '#ffffff', label=f'Bézier Curve (method = {method}, N = {N}, L = {L:.03})')

    # ...
    ax.add_collection3d(mpl_toolkits.mplot3d.art3d.Poly3DCollection(Bounding_Box, linewidths=1.5, edgecolors='#ffd8b2', 
                                                                    facecolors = '#ffffff', alpha=0.01))

    # Set parameters of the graph (plot).
    ax.set_title(f'Bézier Curve Interpolation in {P.shape[1]}-Dimensional Space', fontsize=25, pad=50.0)
    #   Set the x ticks.
    ax.set_xticks(np.arange(np.min(P[:, 0]) - 0.5, np.max(P[:, 0]) + 1.0, 0.5))
    #   Set the y ticks.
    ax.set_yticks(np.arange(np.min(P[:, 1]) - 0.5, np.max(P[:, 1]) + 1.0, 0.5))
    #   Set the z ticks.
    ax.set_zticks(np.arange(np.min(P[:, 2]) - 0.5, np.max(P[:, 2]) + 1.0, 0.5))
    #   Limits.
    ax.set_xlim(np.minimum.reduce(B_Cls.P[:, 0]) - 0.5, np.maximum.reduce(B_Cls.P[:, 0]) + 1.0)
    ax.xaxis.pane.set_color((1.0, 1.0, 1.0, 1.0))
    ax.set_ylim(np.minimum.reduce(B_Cls.P[:, 1]) - 0.5, np.maximum.reduce(B_Cls.P[:, 1]) + 1.0)
    ax.yaxis.pane.set_color((1.0, 1.0, 1.0, 1.0))
    ax.set_zlim(np.minimum.reduce(B_Cls.P[:, 2]) - 0.5, np.maximum.reduce(B_Cls.P[:, 2]) + 1.0)
    ax.zaxis.pane.set_color((1.0, 1.0, 1.0, 1.0))
    #   Label.
    ax.set_xlabel(r'x-axis in meters', fontsize=15, labelpad=10); ax.set_ylabel(r'y-axis in meters', fontsize=15, labelpad=10) 
    ax.set_zlabel(r'z-axis in meters', fontsize=15, labelpad=10) 
    #   Set parameters of the visualization.
    ax.xaxis._axinfo['grid'].update({'linewidth': 0.15, 'linestyle': '--'})
    ax.yaxis._axinfo['grid'].update({'linewidth': 0.15, 'linestyle': '--'})
    ax.zaxis._axinfo['grid'].update({'linewidth': 0.15, 'linestyle': '--'})
    # Get handles and labels for the legend.
    handles, labels = plt.gca().get_legend_handles_labels()
    #   Add a bounding box legend.
    handles.append(pat.Rectangle(xy = (0.0, 0.02), width = 0.0, height = 0.0, facecolor = 'none',
                                edgecolor = '#ffd8b2', linewidth = 1.5))
    labels.append('Bézier Curve Bounding Box')
    # Remove duplicate labels.
    legend = dict(zip(labels, handles))
    # Show the labels (legends) of the graph.
    ax.legend(legend.values(), legend.keys(), fontsize=10.0)

    # Show the result.
    plt.show()

if __name__ == "__main__":
    sys.exit(main())