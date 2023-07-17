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
# Custom Script:
#   ../Lib/Interpolation/Bezier/Core
import Lib.Interpolation.Utilities as Utilities

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
# Save the data to a file.
CONST_SAVE_DATA = False

def main():
    """
    Description:
        A program for visualization of n-th degree B-spline basis functions.
    """

    # Locate the path to the project folder.
    project_folder = os.getcwd().split('Parametric_Curves')[0] + 'Parametric_Curves'

    # Input control points {P} in two-dimensional space.
    P = np.array([[1.00,  0.00], 
                  [2.00, -0.75], 
                  [3.00, -2.50], 
                  [3.75, -1.25], 
                  [4.00,  0.75], 
                  [5.00,  1.00]], dtype=np.float32)

    # Generate a normalized vector of knots from the selected parameters 
    # using the chosen method.
    t = Utilities.Generate_Knot_Vector(CONST_B_SPLINE['n'], P, CONST_B_SPLINE['method'])

    # The value of the time must be within the interval of the knot vector: 
    #   t[0] <= x <= t[-1]
    x = np.linspace(t[0], t[-1], CONST_B_SPLINE['N'])

    # Set the parameters for the scientific style.
    plt.style.use(['science'])

    # Create a figure.
    _, ax = plt.subplots()

    # Visualization of n-th degree B-spline basis functions.
    for j in range(P.shape[0]):
        B_in = np.zeros(x.shape, dtype=x.dtype)
        for i, x_i in enumerate(x):
            B_in[i] = Utilities.Basic_Function(j, CONST_B_SPLINE['n'], t, x_i)
        ax.plot(x, B_in, '-', linewidth=1.0, label=r'$B_{(%d, %d)}(x)$' % (j, CONST_B_SPLINE['n']))

    # Visualization of the normalized vector of knots.
    ax.plot(t, t.shape[0] * [0.0],'o', color='#8d8d8d', linewidth=1.0, markersize = 8.0, 
            markeredgewidth = 4.0, markerfacecolor = '#ffffff', label=r'Normalized Knot Vector')

    # Set parameters of the graph (plot).
    ax.set_title(f"B-spline Basis Functions of the {CONST_B_SPLINE['n']}-th Degree", fontsize=25, pad=25.0)
    #   Set the x ticks.
    ax.set_xticks(np.arange(t[0] - 0.1, t[-1] + 0.1, 0.1))
    #   Set the y ticks.
    ax.set_yticks(np.arange(t[0] - 0.1, t[-1] + 0.1, 0.1))
    #   Label
    ax.set_xlabel(r'x', fontsize=15, labelpad=10); ax.set_ylabel(r'$B_{(i, %d)}(x)$' % CONST_B_SPLINE['n'], fontsize=15, labelpad=10) 
    #   Set parameters of the visualization.
    ax.grid(which='major', linewidth = 0.15, linestyle = '--')
    # Get handles and labels for the legend.
    handles, labels = plt.gca().get_legend_handles_labels()
    # Remove duplicate labels.
    legend = dict(zip(labels, handles))
    # Show the labels (legends) of the graph.
    ax.legend(legend.values(), legend.keys(), fontsize=10.0)

    if CONST_SAVE_DATA == True:
        # Set the full scree mode.
        plt.get_current_fig_manager().full_screen_toggle()

        # Save the results.
        plt.savefig(f'{project_folder}/images/B_Spline/Basic_Functions.png', format='png', dpi=300)
    else:
        # Show the result.
        plt.show()

if __name__ == "__main__":
    sys.exit(main())