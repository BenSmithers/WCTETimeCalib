"""
    Tries to reconstruct the location of each flash in a series of flashes
"""

import os 
from WCTECalib.fitting import fit_hits, get_pmt_positions
from WCTECalib.utils import N_WATER,C, get_color, set_axes_equal
import json 
import numpy as np
from math import sqrt 
from tqdm import tqdm 
import pandas as pd 
from scipy.optimize import basinhopping 

import matplotlib.pyplot as plt 
DEBUG = False

offset_dict = pd.read_csv(
    os.path.join(os.path.dirname(__file__),
    "..",
    "data",
    "calculated_offsets_lbmc.csv")
)


infile = os.path.join(
    os.path.dirname(__file__),
    "..",
    "data",
    "wcsim_offset_swing.json"
)
all_pos = get_pmt_positions()
all_off =  np.array(offset_dict["calc_offset"])

all_radii = []

#simulation result
_obj = open(infile, 'rt')
data = json.load(_obj)
_obj.close()

n_flash = len(data["times"])



for flash_id in tqdm(range(n_flash)):

    _ids = np.array(data["pmtid"][flash_id])+1
    _t_meas= np.array(data["times"][flash_id])
    

    if False:
        window = 20
        options= {
            "eps":1,
            "gtol":1e-5,
        }
        def this_met(params):
            mask = np.abs(_t_meas - params[0])<window 
            return -1*np.sum(mask.astype(int))
        x0 = [np.nanmean(_t_meas)]
        hitres = basinhopping(this_met, x0=x0, niter=10, minimizer_kwargs={"options":options}).x 
        if this_met(hitres)==0:
            continue

        mask = np.abs(_t_meas - hitres[0])<window 
        _ids = _ids[mask]
        _t_meas = _t_meas[mask]

    these_data = np.array([_ids, _t_meas]).T 
    these_data = np.array(sorted(these_data, key=lambda x:x[1] ))

    ids = []
    t_meas = []
    for entry in these_data:
        if entry[0] not in ids:
            ids.append(entry[0])
            t_meas.append(entry[1])
    t_meas = np.array(t_meas)
    ids = np.array(ids).astype(int)




    vector = fit_hits(ids, t_meas)
    all_radii.append(sqrt(np.sum(vector[:3]**2)))

    if DEBUG:
        these_pos = all_pos[ids -1]
        distances = np.sqrt(np.sum((all_pos - vector[0:3])**2, axis=1))
        times = (1e9)*distances*N_WATER/C 

        offset_adj = t_meas + all_off[ids - 1] - vector[3]
        print("{} - {} mean {}".format(min(offset_adj), max(offset_adj), np.mean(offset_adj)))


        ftime = vector[3] 
        mtime = -10
        matime = ftime+30
        colors = get_color( (offset_adj-mtime)/(40), 1)

        ax = plt.axes(projection="3d")
        ax.scatter(these_pos.T[0], these_pos.T[1], these_pos.T[2], color=colors)
        #ax.plot(xs, ys, zs, 'bo')
        ax.plot(vector[0], vector[1], vector[2], 'rd')

        ax.set_xlabel("X [m]")
        ax.set_ylabel("Y [m]")
        ax.set_zlabel("Z [m]")
        set_axes_equal(ax)
        plt.show()


plt.plot(range(len(all_radii)), all_radii, marker='.', ls='')
plt.xlabel("Flash No",size=14)
plt.ylabel("Reco Off-Center Dist [m]",size=14)
plt.savefig("./plots/swinging_cds_ball.png",dpi=400)
plt.show()

all_radii = np.array(all_radii)*1000

bins = np.linspace(0, 100, 101)
bdat = np.histogram(all_radii, bins)[0]

plt.stairs(bdat, bins)
plt.xlabel("Reco Off-Center Dist [mm]",size=14)
plt.show()
