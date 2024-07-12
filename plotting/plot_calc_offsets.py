from mpl_toolkits import mplot3d
from WCTECalib.utils import set_axes_equal, get_color
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
import pandas as pd 
import numpy as np 
import os 

offsets = pd.read_csv(os.path.join(os.path.dirname(__file__), "..","data","calculated_offsets.csv"))

times = offsets["calc_offset"]


colors = get_color( (times-np.min(times))/(np.max(times)-np.min(times)), 1)


fig = plt.figure()
ax = plt.axes(projection="3d")
ax.scatter(offsets["X"], offsets["Y"], offsets["Z"], color=colors)
ax.set_xlabel("X [m]")
ax.set_ylabel("Y [m]")
ax.set_zlabel("Z [m]")
set_axes_equal(ax)
plt.savefig("./plots/offsets_corrected.png", dpi=400)
plt.show()
