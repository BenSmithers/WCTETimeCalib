from mpl_toolkits import mplot3d
from WCTECalib.utils import set_axes_equal, get_color
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
import pandas as pd 
import numpy as np 
import os 

ERRORS = False 


true = pd.read_csv(os.path.join(os.path.dirname(__file__), "..","data","offsets.csv"))

offsets = pd.read_csv(os.path.join(os.path.dirname(__file__), "..","data","calculated_offsets.csv"))

times = offsets["calc_offset"]
true_offsets = true["offsets"]

if ERRORS:
    times = times - (true_offsets[0]-true_offsets )


print("{} - {}".format(min(times), max(times))) 
if ERRORS:
    colors = get_color( times+0.25, 1)
else:
    colors = get_color( (times -min(times))/(max(times)-min(times)), 1)


fig = plt.figure()
ax = plt.axes(projection="3d")
ax.scatter(offsets["X"], offsets["Y"], offsets["Z"], color=colors)
ax.set_xlabel("X [m]")
ax.set_ylabel("Y [m]")
ax.set_zlabel("Z [m]")
set_axes_equal(ax)
plt.savefig("./plots/offsets_corrected.png", dpi=400)
plt.show()


if ERRORS:
    bins = np.linspace(-1, 1, 100)
    histo = np.histogram(times, bins)
    plt.stairs(histo[0], bins)
    plt.xlabel("Offset Error [ns]", size=14)
    plt.ylabel("Counts")
    plt.tight_layout()
    plt.show()

