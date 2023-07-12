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
# Custom Script:
#   ../Lib/Interpolation/B_Spline/Core
import Lib.Interpolation.B_Spline.Core as B_Spline

def main():
    # ...
    #   ...
    n = 3; N = 100; method = 'Chord-Length'
    #   ...
    P = np.array([[1.00,  0.00, -1.00], 
                  [2.00, -0.75,  0.50], 
                  [3.00, -2.50,  1.00], 
                  [3.75, -1.25, -0.50], 
                  [4.00,  0.75,  1.50], 
                  [5.00,  1.00, -1.50]])

    # ...
    S_Cls_1 = B_Spline.B_Spline_Cls(n, P, method, N)
    # ...
    S = S_Cls_1.Interpolate()

    # ...
    S_noise = np.zeros(S.shape)
    for i, S_i in enumerate(S):
        S_noise[i, :] = S_i + np.random.uniform((-1) * np.random.uniform(0.05, 0.20), 
                                              np.random.uniform(0.05, 0.20), S.shape[1])
    S_noise[0] = P[0]; S_noise[-1] = P[-1]

    # ...
    S_Cls_2 = B_Spline.B_Spline_Cls(n, S_noise, method, N)
    # ...
    S_Cls_optimized = S_Cls_2.Optimize_Control_Points(P.shape[0])
    S_optimized = S_Cls_optimized.Interpolate()

    # ...
    L = S_Cls_optimized.Get_Arc_Length()

    # Set the parameters for the scientific style.
    plt.style.use(['science'])

    # Create a figure.
    figure = plt.figure()
    ax = figure.add_subplot(projection='3d')

    # ... 
    ax.plot(S_noise[:, 0], S_noise[:, 1], S_noise[:, 2], 'o', color='#e7e7e7', linewidth=1.0, markersize = 8.0, 
            markeredgewidth = 4.0, markerfacecolor = '#ffffff', label='Noisy Control Points')
    # ...
    ax.plot(S_Cls_optimized.P[:, 0], S_Cls_optimized.P[:, 1], S_Cls_optimized.P[:, 2], 'o--', color='#8ca8c5', linewidth=1.0, markersize = 8.0, 
            markeredgewidth = 4.0, markerfacecolor = '#ffffff', label='Optimized Control Points')
    # ...
    ax.plot(S_optimized[:, 0], S_optimized[:, 1], S_optimized[:, 2], '.-', color='#ffbf80', linewidth=1.5, markersize = 8.0, 
            markeredgewidth = 2.0, markerfacecolor = '#ffffff', label=f'B-Spline (n = {n}, N = {N}, L = {L:.03})')

    # Set parameters of the graph (plot).
    ax.set_title(f'B-Spline Interpolation in {P.shape[1]}-Dimensional Space', fontsize=25, pad=50.0)
    #   Set the x ticks.
    ax.set_xticks(np.arange(np.min(P[:, 0]) - 0.5, np.max(P[:, 0]) + 1.0, 0.5))
    #   Set the y ticks.
    ax.set_yticks(np.arange(np.min(P[:, 1]) - 0.5, np.max(P[:, 1]) + 1.0, 0.5))
    #   Set the z ticks.
    ax.set_zticks(np.arange(np.min(P[:, 2]) - 0.5, np.max(P[:, 2]) + 1.0, 0.5))
    #   Limits.
    ax.set_xlim(np.minimum.reduce(S_Cls_1.P[:, 0]) - 0.5, np.maximum.reduce(S_Cls_1.P[:, 0]) + 1.0)
    ax.xaxis.pane.set_color((1.0, 1.0, 1.0, 1.0))
    ax.set_ylim(np.minimum.reduce(S_Cls_1.P[:, 1]) - 0.5, np.maximum.reduce(S_Cls_1.P[:, 1]) + 1.0)
    ax.yaxis.pane.set_color((1.0, 1.0, 1.0, 1.0))
    ax.set_zlim(np.minimum.reduce(S_Cls_1.P[:, 2]) - 0.5, np.maximum.reduce(S_Cls_1.P[:, 2]) + 1.0)
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
    # Remove duplicate labels.
    legend = dict(zip(labels, handles))
    # Show the labels (legends) of the graph.
    ax.legend(legend.values(), legend.keys(), fontsize=10.0)

    # Show the result.
    plt.show()

if __name__ == "__main__":
    sys.exit(main())