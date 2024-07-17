from WCTECalib.times import sample_leds
from WCTECalib.geometry import get_pmt_positions, get_led_positions, get_led_dirs
from WCTECalib.utils import get_color, set_axes_equal

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
from mpl_toolkits import mplot3d

import numpy as np

# 16 mPMTs per loop
# 3 LEDs per mPMT 
# step in incremends of 48 
start = (9+12)*3  + 16*3

colors = ["green", "blue", "red", "orange", 'purple', 'black']

fig = plt.figure()
ax = plt.axes(projection="3d")
for i in range(4):
    which = start + i*4*3 

    led_ar_pos = get_led_positions([which,])
    led_dir = get_led_dirs([which,]).T
    led_pos = led_ar_pos[0]
    ids, times, evts = sample_leds(which, mu=100)

    positions = get_pmt_positions(ids).T 
    color = get_color(evts, np.max(evts), 'inferno')

    ax.quiver(led_pos[0], led_pos[1],led_pos[2], led_dir[0], led_dir[1], led_dir[2],color='black', length=0.5, normalize=True)
    ax.scatter(positions[0], positions[1], positions[2], color=color) # colors[i])
    break

ax.set_xlabel("X [m]")
ax.set_ylabel("Y [m]")
ax.set_zlabel("Z [m]")
set_axes_equal(ax)
plt.show()
