from mpl_toolkits import mplot3d
from WCTECalib.utils import set_axes_equal
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
import pandas as pd 
import numpy as np 
import os 

offsets = pd.read_csv(os.path.join(os.path.dirname(__file__), "..","data","offsets.csv"))

times = offsets["offsets"]

def get_color(n, colormax=3.0, cmap="RdBu"):
    """
        Discretize a colormap. Great getting nice colors for a series of trends on a single plot! 
    """
    this_cmap = plt.get_cmap(cmap)
    return this_cmap(n/colormax)

colors = get_color( (times-np.min(times))/(np.max(times)-np.min(times)), 1)



fig = plt.figure()
ax = plt.axes(projection="3d")
ax.scatter(offsets["X"], offsets["Y"], offsets["Z"], color=colors)
ax.set_xlabel("X [m]")
ax.set_ylabel("Y [m]")
ax.set_zlabel("Z [m]")
set_axes_equal(ax)
plt.savefig("./plots/offsets_uncorrected.png", dpi=400)
plt.show()
