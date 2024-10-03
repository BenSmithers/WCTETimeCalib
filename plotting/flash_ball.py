from WCTECalib.times import sample_balltime, second, BALL_ERR
from WCTECalib import df, N_CHAN, get_pmt_positions, N_MPMT
from WCTECalib.utils import ball_pos, C, N_WATER, NOISE_SCALE, convert_to_2d_offset
from WCTECalib.utils import set_axes_equal, get_color


import numpy as np 


import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
from mpl_toolkits import mplot3d


noise = 0.5*second
ball_pos_err = np.random.randn(3)*0.0

all_ids = np.array(range(N_MPMT*N_CHAN))

ids, t_meas, npe = sample_balltime(noise=noise, ball=ball_pos, ball_pos_noise=False, diff_err=False, mu=0.3)

positions_keep = get_pmt_positions(ids).T


keeps = np.zeros(N_MPMT*N_CHAN, dtype=int)
keeps[ids] = 1

nokeep = np.logical_not(keeps)

positions_discard = get_pmt_positions(all_ids[nokeep]).T

fig = plt.figure()
ax = plt.axes(projection="3d")

colorkeep = get_color(npe, 3, "inferno")
colorreject = get_color(np.zeros_like(positions_discard[0]), 3, "inferno")
#ax.scatter(positions_discard[0], positions_discard[1], positions_discard[2], color='gray')
ax.scatter(positions_keep[0], positions_keep[1], positions_keep[2], color=colorkeep)
ax.scatter(positions_discard[0], positions_discard[1], positions_discard[2], color=colorreject)
ax.set_xlabel("X [m]")
ax.set_ylabel("Y [m]")
ax.set_zlabel("Z [m]")
plt.show()