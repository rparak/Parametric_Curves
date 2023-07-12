# System (Default)
import sys
#   Add access if it is not in the system path.
if '../' + 'src' not in sys.path:
    sys.path.append('../..')
# Numpy (Array computing) [pip3 install numpy]
import numpy as np
# OS (Operating system interfaces)
import os
# SciencePlots (Matplotlib styles for scientific plotting) [pip3 install SciencePlots]
import scienceplots
# Matplotlib (Visualization) [pip3 install matplotlib]
import matplotlib.pyplot as plt
import matplotlib.patches as pat
# Custom Script:
#   ../Lib/Interpolation/Bezier/Core
import Lib.Interpolation.Bezier.Core as Bezier

def main():
    # ...
    #   ...
    method = 'Explicit'; N = 100
    #   ...
    P = np.array([[1.00,  0.00], 
                  [2.00, -0.75], 
                  [3.00, -2.50], 
                  [3.75, -1.25], 
                  [4.00,  0.75], 
                  [5.00,  1.00]])

    # ...
    B_Cls = Bezier.Bezier_Cls(method, P, N)
    # ...
    B = B_Cls.Interpolate()
    # ...
    L = B_Cls.Get_Arc_Length()
    # ...
    B_Bounding_Box = B_Cls.Get_Bounding_Box_Parameters('Interpolated-Points')

    # Set the parameters for the scientific style.
    plt.style.use(['science'])

    # Create a figure.
    _, ax = plt.subplots()

    # ...
    ax.plot(P[:, 0], P[:, 1], 'o--', color='#d0d0d0', linewidth=1.0, markersize = 8.0, 
            markeredgewidth = 4.0, markerfacecolor = '#ffffff', label='Control Points')
    # ...
    ax.plot(B[:, 0], B[:, 1], '.-', color='#ffbf80', linewidth=1.5, markersize = 8.0, 
            markeredgewidth = 2.0, markerfacecolor = '#ffffff', label=f'Bézier Curve (method = {method}, N = {N}, L = {L:.03})')
    
    # ...
    Bounding_Box_Interpolated_Points = pat.Rectangle(xy = (B_Bounding_Box['x_min'], B_Bounding_Box['y_min']), width = B_Bounding_Box['x_max'] - B_Bounding_Box['x_min'],
                                                     height = B_Bounding_Box['y_max'] -  B_Bounding_Box['y_min'], facecolor = 'none', edgecolor = '#ffd8b2', linewidth = 1.5, label='Bézier Curve Bounding Box')
    ax.add_patch(Bounding_Box_Interpolated_Points)

    # Set parameters of the graph (plot).
    ax.set_title(f'Bézier Curve Interpolation in {P.shape[1]}-Dimensional Space', fontsize=25, pad=50.0)
    #   Set the x ticks.
    ax.set_xticks(np.arange(np.min(P[:, 0]) - 0.5, np.max(P[:, 0]) + 1.0, 0.5))
    #   Set the y ticks.
    ax.set_yticks(np.arange(np.min(P[:, 1]) - 0.5, np.max(P[:, 1]) + 1.0, 0.5))
    #   Label
    ax.set_xlabel(r'x-axis in meters', fontsize=15, labelpad=10); ax.set_ylabel(r'y-axis in meters', fontsize=15, labelpad=10) 
    #   Set parameters of the visualization.
    ax.grid(which='major', linewidth = 0.25, linestyle = '--')
    # Get handles and labels for the legend.
    handles, labels = plt.gca().get_legend_handles_labels()
    # Remove duplicate labels.
    legend = dict(zip(labels, handles))
    # Show the labels (legends) of the graph.
    ax.legend(legend.values(), legend.keys(), fontsize=10.0)

    # Show the result.
    plt.show()

if __name__ == "__main__":
    sys.exit(main())