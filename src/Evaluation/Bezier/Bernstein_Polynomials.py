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
# Custom Lib.:
#   ../Lib/Interpolation/Bezier/Core
import Lib.Interpolation.Utilities as Utilities

"""
Description:
    Initialization of constants.
"""
# Bezier curve interpolation parameters.
#   N: The number of points to be generated in the interpolation function.
CONST_BEZIER_CURVE = {'N': 100}
# Save the data to a file.
CONST_SAVE_DATA = False

def main():
    """
    Description:
        A program for visualization of Bernstein basis polynomials functions of degree n.
    """
    
    # Locate the path to the project folder.
    project_folder = os.getcwd().split('Parametric_Curves')[0] + 'Parametric_Curves'

    # Input control points {P} in two-dimensional space.
    P = np.array([[1.00,  0.00], 
                  [2.00, -0.75], 
                  [3.00, -2.50], 
                  [3.75, -1.25], 
                  [4.00,  0.75], 
                  [5.00,  1.00]], dtype=np.float64)
    
    # The value of the time must be within the interval: 
    #   0.0 <= x <= 1.0
    x = np.linspace(Utilities.CONST_T_0, Utilities.CONST_T_1, CONST_BEZIER_CURVE['N'])

    # Degree of a polynomial.
    n = P.shape[0] - 1

    # Set the parameters for the scientific style.
    plt.style.use(['science'])

    # Create a figure.
    _, ax = plt.subplots()

    # Visualization of Bernstein basis polynomials functions of degree n.
    for i in range(P.shape[0]):
        B_in = Utilities.Bernstein_Polynomial(i, n, x)
        ax.plot(x, B_in, '-', linewidth=1.0, label=r'$B_{(%d, %d)}(x)$' % (i, n))

    # Set parameters of the graph (plot).
    ax.set_title(f'Bernstein Basis Polynomials of the Degree {n}', fontsize=25, pad=25.0)
    #   Set the x ticks.
    ax.set_xticks(np.arange(Utilities.CONST_T_0 - 0.1, Utilities.CONST_T_1 + 0.1, 0.1))
    #   Set the y ticks.
    ax.set_yticks(np.arange(Utilities.CONST_T_0 - 0.1, Utilities.CONST_T_1 + 0.1, 0.1))
    #   Label
    ax.set_xlabel(r'x', fontsize=15, labelpad=10); ax.set_ylabel(r'$B_{(i, %d)}(x)$' % n, fontsize=15, labelpad=10) 
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
        plt.savefig(f'{project_folder}/images/Bezier/Bernstein_Polynomials.png', format='png', dpi=300)
    else:
        # Show the result.
        plt.show()

if __name__ == "__main__":
    sys.exit(main())
