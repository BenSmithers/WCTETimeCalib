from mpl_toolkits import mplot3d

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
import pandas as pd 
import numpy as np 
import os 

offsets = pd.read_csv(os.path.join(
    os.path.dirname(__file__), 
    "..",
    "data",
    "offset_dataframe.csv"))

ball_result =  pd.read_csv(os.path.join(
    os.path.dirname(__file__),
    "..",
    "data",
    "perturbed_ball.csv")
)

pulse_times = np.array(ball_result["pulse_times"]) - np.array(offsets["offset"])


def get_color(n, colormax=3.0, cmap="jet"):
    """
        Discretize a colormap. Great getting nice colors for a series of trends on a single plot! 
    """
    this_cmap = plt.get_cmap(cmap)
    return this_cmap(n/colormax)

colors = get_color( (pulse_times-np.min(pulse_times))/(np.max(pulse_times)-np.min(pulse_times)), 1)


fig = plt.figure()
ax = plt.axes(projection="3d")
ax.scatter(offsets["X"], offsets["Y"], offsets["Z"], color=colors)
plt.show()
