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
import matplotlib.patches as pat
# Custom Script:
#   ../Lib/Interpolation/B_Spline/Core
import Lib.Interpolation.B_Spline.Core as B_Spline

"""
Description:
    Initialization of constants.
"""
# B-Spline interpolation parameters.
CONST_B_SPLINE = {'n': 3, 'N': 100, 'method': 'Chord-Length'}
#   Visibility of the bounding box of the interpolated curve.
CONST_BOUNDING_BOX_VISIBILITY = False

def main():
    """
    Description:
        ..
    """
        
    # Input control points {P} in two-dimensional space.
    P = np.array([[1.00,  0.00], 
                  [2.00, -0.75], 
                  [3.00, -2.50], 
                  [3.75, -1.25], 
                  [4.00,  0.75], 
                  [5.00,  1.00]], dtype=np.float32)

    # ...
    S_Cls_1 = B_Spline.B_Spline_Cls(CONST_B_SPLINE['n'], CONST_B_SPLINE['method'], P, 
                                    CONST_B_SPLINE['N'])
    # ...
    S = S_Cls_1.Interpolate()

    # ...
    S_noise = np.zeros(S.shape)
    for i, S_i in enumerate(S):
        S_noise[i, :] = S_i + np.random.uniform((-1) * np.random.uniform(0.05, 0.20), 
                                              np.random.uniform(0.05, 0.20), S.shape[1])
    S_noise[0] = P[0]; S_noise[-1] = P[-1]

    # ...
    S_Cls_2 = B_Spline.B_Spline_Cls(CONST_B_SPLINE['n'], CONST_B_SPLINE['method'], S_noise, 
                                    CONST_B_SPLINE['N'])
    # ...
    S_Cls_optimized = S_Cls_2.Optimize_Control_Points(P.shape[0])
    S_optimized = S_Cls_optimized.Interpolate()

    # Obtain the arc length L(t) of the general parametric curve.
    L = S_Cls_optimized.Get_Arc_Length()

    # Set the parameters for the scientific style.
    plt.style.use(['science'])

    # Create a figure.
    _, ax = plt.subplots()

    # ... 
    ax.plot(S_noise[:, 0], S_noise[:, 1], 'o', color='#e7e7e7', linewidth=1.0, markersize = 8.0, 
            markeredgewidth = 4.0, markerfacecolor = '#ffffff', label='Noisy Control Points')
    # ...
    ax.plot(S_Cls_optimized.P[:, 0], S_Cls_optimized.P[:, 1], 'o--', color='#8ca8c5', linewidth=1.0, markersize = 8.0, 
            markeredgewidth = 4.0, markerfacecolor = '#ffffff', label='Optimized Control Points')
    # ...
    ax.plot(S_optimized[:, 0], S_optimized[:, 1], '.-', color='#ffbf80', linewidth=1.5, markersize = 8.0, 
            markeredgewidth = 2.0, markerfacecolor = '#ffffff', label=f'B-Spline (n = {S_Cls_optimized.n}, N = {S_Cls_optimized.N}, L = {L:.03})')
    
    # Visibility of the bounding box of the interpolated curve.
    if CONST_BOUNDING_BOX_VISIBILITY == True:
        # Obtain the bounding parameters (min, max) of the general parametric curve.
        S_Bounding_Box = S_Cls_optimized.Get_Bounding_Box_Parameters('Interpolated-Points')

        # Create a primitive two-dimensional object (Rectangle -> Bounding-Box) with additional properties.
        Bounding_Box_Interpolated_Points = pat.Rectangle(xy = (S_Bounding_Box['x_min'], S_Bounding_Box['y_min']), width = S_Bounding_Box['x_max'] - S_Bounding_Box['x_min'],
                                                         height = S_Bounding_Box['y_max'] -  S_Bounding_Box['y_min'], facecolor = 'none', edgecolor = '#ffd8b2', linewidth = 1.5, 
                                                         label='B-Spline Bounding Box')
        ax.add_patch(Bounding_Box_Interpolated_Points)

    # Set parameters of the graph (plot).
    ax.set_title(f'B-Spline Interpolation in {P.shape[1]}-Dimensional Space', fontsize=25, pad=25.0)
    #   Set the x ticks.
    ax.set_xticks(np.arange(np.min(S_Cls_1.P[:, 0]) - 0.5, np.max(S_Cls_1.P[:, 0]) + 1.0, 0.5))
    #   Set the y ticks.
    ax.set_yticks(np.arange(np.min(S_Cls_1.P[:, 1]) - 0.5, np.max(S_Cls_1.P[:, 1]) + 1.0, 0.5))
    #   Label
    ax.set_xlabel(r'x-axis in meters', fontsize=15, labelpad=10); ax.set_ylabel(r'y-axis in meters', fontsize=15, labelpad=10) 
    #   Set parameters of the visualization.
    ax.grid(which='major', linewidth = 0.15, linestyle = '--')
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