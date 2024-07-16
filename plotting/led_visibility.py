from math import pi
from WCTECalib.geometry import get_pmts_visible, get_led_positions
from WCTECalib.utils import set_axes_equal, get_color

import numpy as np 

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
from mpl_toolkits import mplot3d
import os 
import pandas as pd

offsets = pd.read_csv(os.path.join(os.path.dirname(__file__), "..","data","geometry.csv"))
LED_NO = 0.0

led_ar_pos = get_led_positions([LED_NO,])
led_pos = led_ar_pos[0]

keep  = np.array(get_pmts_visible(LED_NO))
#idlist= idlist.astype(int)

#keep = np.zeros_like(offsets["X"])
#keep[idlist] = 1
#keep = keep.astype(bool)

colors = get_color( keep, 1, "RdYlGn")

alphas = np.ones_like(keep)
alphas[np.logical_not(keep)] = 0.1


fig = plt.figure()
ax = plt.axes(projection="3d")
ax.scatter(offsets["X"], offsets["Y"], offsets["Z"],zorder=0, color=colors,)
ax.scatter(led_pos[0], led_pos[1], led_pos[2], color='green', zorder=10, s=100, alpha=1)
ax.scatter([],[],[], color='green', label="Visible")
plt.legend()
ax.set_xlabel("X [m]")
ax.set_ylabel("Y [m]")
ax.set_zlabel("Z [m]")
set_axes_equal(ax)
plt.show()
