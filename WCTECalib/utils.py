
import numpy as  np
C = 2.99e8 # m/s

ns = 1 
second = 1e9
mm = 1e-3 
NOISE_SCALE = 1.5*ns 
OFFSET_SCALE = 60*ns 
PMT_OFF = 20*ns
DIFFUSER_ERR = 1*ns
BALL_ERR = 2*mm 
SAMPLE_SIZE = 8*ns

ABS_LEN = 20# meters 


N_WATER = 1.33 
ball_pos = np.array([0,0.0, 0])

def convert_to_2d_offset(offset_dict, as_dict=True ):
    """
        Returns a 2D array for the relative offset between any combination of PMTs
    """
    if as_dict:
        calculated_offset = np.array(offset_dict["calc_offset"])
    else:
        calculated_offset=offset_dict
    rotated = np.reshape(calculated_offset,(len(calculated_offset), 1))
    return rotated - rotated.T


# some random values
CABLE_DELAY = np.array([
    4.0923004 ,   0.45297347, -13.00278037,   3.11316735,
    -4.55179197,  -3.29802082,   6.34444509,  -6.90585025,
    1.78342475,  -1.09273518,   2.79274299,  -6.35754776,
    0.65196117, -10.57912301,   9.44017319, -10.28843254,
    3.49498734,  -0.76563351,  -3.74257867
]) 
CABLE_DELAY += np.min(CABLE_DELAY)

def set_axes_equal(ax):
    '''Make axes of 3D plot have equal scale so that spheres appear as spheres,
    cubes as cubes, etc..  This is one possible solution to Matplotlib's
    ax.set_aspect('equal') and ax.axis('equal') not working for 3D.

    Input
    ax: a matplotlib axis, e.g., as output from plt.gca().
    
    FROM: https://stackoverflow.com/questions/13685386/matplotlib-equal-unit-length-with-equal-aspect-ratio-z-axis-is-not-equal-to
    '''

    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)

    # The plot bounding box is a sphere in the sense of the infinity
    # norm, hence I call half the max range the plot radius.
    plot_radius = 0.4*max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])

import matplotlib.pyplot as plt 

def get_color(n, colormax=3.0, cmap="RdBu"):
    """
        Discretize a colormap. Great getting nice colors for a series of trends on a single plot! 
    """
    
    this_cmap = plt.get_cmap(cmap)
    return this_cmap(n/colormax)