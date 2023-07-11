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
#   ../Lib/Interpolation/B_Spline/Core
import Lib.Interpolation.B_Spline.Core as B_Spline

def main():
    # ...
    #   ...
    n = 3; N = 100
    #   ...
    P = np.array([[1.00,  0.00], 
                  [2.00, -0.75], 
                  [3.00, -2.50], 
                  [3.75, -1.25], 
                  [4.00,  0.75], 
                  [5.00,  1.00]])

    # ...
    S_Cls = B_Spline.B_Spline_Cls(n, P, 'Chord-Length', N)
    # ...
    S = S_Cls.Interpolate()
    # ...
    L = S_Cls.Get_Arc_Length()
    # ...
    P_Bounding_Box = S_Cls.Get_Bounding_Box_Parameters('Control-Points')
    S_Bounding_Box = S_Cls.Get_Bounding_Box_Parameters('Interpolated-Points')

    # Set the parameters for the scientific style.
    plt.style.use(['science'])

    # Create a figure.
    _, ax = plt.subplots()

    # ...
    ax.plot(P[:, 0], P[:, 1], 'o--', color='#d0d0d0', linewidth=1.0, markersize = 8.0, 
            markeredgewidth = 4.0, markerfacecolor = '#ffffff', label='Control Points')
    # ...
    ax.plot(S[:, 0], S[:, 1], '.-', color='#ffbf80', linewidth=1.5, markersize = 8.0, 
            markeredgewidth = 2.0, markerfacecolor = '#ffffff', label=f'Interpolated Points: n = {n}, N = {N}')
    
    # ...
    Bounding_Box_Control_Points = pat.Rectangle(xy = (P_Bounding_Box['x_min'], P_Bounding_Box['y_min']), width = P_Bounding_Box['x_max'] - P_Bounding_Box['x_min'],
                                                height = P_Bounding_Box['y_max'] -  P_Bounding_Box['y_min'], facecolor = 'none', edgecolor = '#e1e1e1', linewidth = 1.5)
    ax.add_patch(Bounding_Box_Control_Points)
    Bounding_Box_Interpolated_Points = pat.Rectangle(xy = (S_Bounding_Box['x_min'], S_Bounding_Box['y_min']), width = S_Bounding_Box['x_max'] - S_Bounding_Box['x_min'],
                                                     height = S_Bounding_Box['y_max'] -  S_Bounding_Box['y_min'], facecolor = 'none', edgecolor = '#ffd8b2', linewidth = 1.5)
    ax.add_patch(Bounding_Box_Interpolated_Points)

    # Set parameters of the graph (plot).
    ax.set_title('B-Spline Interpolation', fontsize=25, pad=50.0)
    ax.text(((np.min(P[:, 0]) + np.max(P[:, 0]))/2.0), 1.625, f'Arc Length: {L:.03}', fontsize=15, 
            bbox={'facecolor': '#d0d0d0', 'alpha': 0.1, 'pad': 5}, ha='center', va='center')
    #   Set the x ticks.
    ax.set_xticks(np.arange(np.min(P[:, 0]) - 1.0, np.max(P[:, 0]) + 1.0, 0.5))
    #   Set the y ticks.
    ax.set_yticks(np.arange(np.min(P[:, 1]) - 1.0, np.max(P[:, 1]) + 1.0, 0.5))
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