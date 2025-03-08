"""
    Tries to reconstruct the location of each flash in a series of flashes
"""

import os 
from WCTECalib.fitting import fit_hits, get_pmt_positions
from WCTECalib.utils import N_WATER,C, get_color, set_axes_equal
from WCTECalib import df
import json 
import numpy as np
from math import sqrt 
from tqdm import tqdm 
import pandas as pd 
from scipy.optimize import basinhopping 

from event_process import process

import matplotlib.pyplot as plt 
import h5py as h5 
DEBUG = True

offset_dict = pd.read_csv(
    os.path.join(os.path.dirname(__file__),
    "..",
    "data",
    "calculated_offsets_realdata.csv")
)


infile = os.path.join(
    os.path.dirname(__file__),
    "..",
    "data",
    "laserball_realdata_events.h5"
)

all_off =  np.array(offset_dict["calc_offset"])

all_radii = []
xs = []
ys = []
zs = []
data = h5.File(infile)

n_flash = len(data.keys())


reference_id = 25

for _flash_id in tqdm(range(50)):
    flash_id = _flash_id+1
    if flash_id>1000:
        break
    
    ekey ="event{}".format(flash_id)
    if ekey not in data:
        continue
    ids, times, charge = process(data[ekey]) 
    print("{} ids".format(len(ids)))   

    bad_batch = np.array([57, 58, 81.])



    #print("To {} times, ranging {}  to {}".format(len(t_meas), t_meas.min(), t_meas.max()))
    

    bins = np.linspace(np.nanmin(times),np.nanmax(times), 100)
    all_binned = np.histogram(times, bins)[0]
    max_time = bins[np.argmax(all_binned)]

    if np.sum(np.isnan(times).astype(int)) == len(times):
        continue
    if len(times)==0:
        continue
    cut = np.logical_and( times>max_time-5, times<max_time+15 )

    #ids = ids[cut]
    #times = times[cut]
    #charge = charge[cut]
    bins = np.linspace(np.nanmin(times),np.nanmax(times), 100)
    print(np.nanmin(times), np.nanmax(times))
    if len(times)==0:
        continue
#    plt.hist(times, bins=bins)
#    plt.show()


    vector = fit_hits(ids, times)
    all_radii.append(sqrt(np.sum(vector[:3]**2)))
    xs.append(vector[0])
    ys.append(vector[1])
    zs.append(vector[2])

    if DEBUG:

        mpmt_no = ids // 19
        channel = ids % 19

        stacked = np.zeros(len(bins)-1)
        zthis = 100
        print(np.unique(mpmt_no))

        for mpmt in np.unique(mpmt_no):
            binned = np.histogram((times)[mpmt_no==mpmt], bins, )[0] 
            plt.stairs(binned+stacked, bins, zorder= zthis, color=get_color(zthis, 100, "jet"), fill=True)
            zthis -= 1
            stacked += binned
            if zthis<0:
                break

        shift_time = bins[np.argmax(stacked)]

        plt.xlabel("Hit Time [ns]")
        plt.show()

        these_pos = get_pmt_positions(ids)
        distances = np.sqrt(np.sum((these_pos - vector[0:3])**2, axis=1))

        calc_time = (1e9)*distances*N_WATER/C 
        #print("offset time", vector[3])
        offset_adj = times -calc_time - vector[3] 

        colors = times -shift_time
        
        ax = plt.axes(projection="3d")
        scatty = ax.scatter(these_pos.T[0], these_pos.T[1], these_pos.T[2], c=colors,vmin=-5, vmax=5, cmap=plt.cm.RdBu)
        #ax.plot(xs, ys, zs, 'bo')
        ax.plot(vector[0], vector[1], vector[2], 'rd')
        plt.colorbar(scatty, ax=ax)
        ax.set_xlabel("X [m]")
        ax.set_ylabel("Y [m]")
        ax.set_zlabel("Z [m]")
        set_axes_equal(ax)
        plt.show()


#plt.plot(range(len(all_radii)), all_radii, marker='.', ls='')
plt.plot(range(len(all_radii)), xs, marker='.', ls='',label="X")
plt.plot(range(len(all_radii)), ys, marker='.', ls='', label="Y")
plt.plot(range(len(all_radii)), zs, marker='.', ls='', label="Z")
plt.legend()
plt.xlabel("Flash No",size=14)
plt.ylabel("Loc [m]",size=14)
plt.savefig("./plots/swinging_cds_ball.png",dpi=400)
plt.show()

all_radii = np.array(all_radii)*1000

bins = np.linspace(0, 100, 101)
bdat = np.histogram(all_radii, bins)[0]

plt.stairs(bdat, bins)
plt.xlabel("Reco Off-Center Dist [mm]",size=14)
plt.show()
