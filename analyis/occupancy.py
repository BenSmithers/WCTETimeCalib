import h5py as h5 
import os 
from tqdm import tqdm 
from event_process import process
import numpy as np 

from WCTECalib.fitting import fit_hits, get_pmt_positions
from WCTECalib.utils import set_axes_equal

import matplotlib.pyplot as plt 

infile = os.path.join(
    os.path.dirname(__file__),
    "..",
    "data",
    "laserball_realdata_events.h5"
)

data = h5.File(infile)

n_flash = len(data.keys())

n_hits = {}

for _flash_id in tqdm(range(n_flash)):
    flash_id = _flash_id+1

    

    ids, times, charge = process(data["event{}".format(flash_id)])    
    
    for ent in ids: 
        if ent in n_hits:
            n_hits[ent] += 1.0/n_flash 
        else:
            n_hits[ent] = 1.0/n_flash

dict_ids = np.array(list(n_hits.keys())).flatten() 
dict_occupancy = np.array(list(n_hits.values())).flatten() 
these_pos = get_pmt_positions(dict_ids)
print("{} - {}".format(np.min(dict_occupancy), np.max(dict_occupancy)))

ax = plt.axes(projection="3d")
scatty = ax.scatter(these_pos.T[0], these_pos.T[1], these_pos.T[2], c=dict_occupancy,vmin=0, vmax=1, cmap=plt.cm.inferno)
#ax.plot(xs, ys, zs, 'bo')
plt.colorbar(scatty, ax=ax)
ax.set_xlabel("X [m]")
ax.set_ylabel("Y [m]")
ax.set_zlabel("Z [m]")
set_axes_equal(ax)

plt.savefig("./plots/occupancy.png",dpi=400)