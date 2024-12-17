import numpy as np
import matplotlib.pyplot as plt 

from mpl_toolkits.mplot3d import axes3d
from mpl_toolkits import mplot3d
import sys 
from tqdm import tqdm 
import h5py as h5 

import json 
from random import choice 
from WCTECalib.utils import C, N_WATER, mm,ball_pos, second
from WCTECalib import df, N_CHAN, get_pmt_positions, N_MPMT

if len(sys.argv)<2:
    raise Exception("Insufficient number of arguments: {}<2".format(len(sys.argv)))

fig = plt.figure()


infile = sys.argv[1]

#simulation result
_obj = open(infile, 'rt')
data = json.load(_obj)
_obj.close()

n_flash = len(data["times"])
reference_id = -1 

for flash_id in range(n_flash):
    # ids in the data files are off by 1 relative to the geo file
    

    _ids = np.array(data["pmtid"][flash_id])+1
    if reference_id==-1:
        reference_id = _ids[0]
    if reference_id not in _ids:
        continue

    
    _t_meas= np.array(data["times"][flash_id])
    print(len(_t_meas), "hits")
    _charge = np.array(data["charge"][flash_id])

    

    these_data = np.array([_ids, _t_meas, _charge]).T 
    these_data = np.array(sorted(these_data, key=lambda x:x[1] ))

    ids = []
    charges = []
    t_meas = []
    for entry in these_data:
        if entry[0] not in ids:
            ids.append(entry[0])
            charges.append(entry[2])
            t_meas.append(entry[1])



    t_meas = np.array(t_meas)
    ids = np.array(ids)
    bin_where = choice(np.argwhere(ids==reference_id))
    

    _t_meas = _t_meas - t_meas[bin_where] 


    # let's plot them 
    ax = plt.axes(projection="3d")
    positions = get_pmt_positions(ids).T 
    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.set_zlabel("Z [m]")
    ax.scatter(positions[0], positions[1], positions[2])
    plt.show()
    plt.clf()