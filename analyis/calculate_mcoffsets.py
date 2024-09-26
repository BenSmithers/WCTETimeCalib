
import os 
import json 
import numpy as np 
from copy import deepcopy
from WCTECalib.utils import C, N_WATER, mm,ball_pos, second
from WCTECalib.alt_geo import df, N_CHAN, get_pmt_positions, N_MPMT
from tqdm import tqdm 
from scipy.signal import find_peaks
from scipy.optimize import minimize
from math import sqrt,log
DEBUG =True

central_ball_loc = ball_pos

outfile = os.path.join(
    os.path.dirname(__file__),
    "..",
    "data",
    "wcsim_offset.json"
)

offsets = np.linspace(-180, 180, 3600)
offset_center = 0.5*(offsets[1:] + offsets[:-1])
binids = np.arange(-0.5, N_CHAN*N_MPMT+0.5, 1)+1 
all_bins = np.zeros(( len(binids)-1, len(offsets)-1))


#simulation result
_obj = open(outfile, 'rt')
data = json.load(_obj)
_obj.close()

n_flash = len(data["times"])

for flash_id  in range(n_flash):
    # ids in the data files are off by 1 relative to the geo file
    _ids = np.array(data["pmtid"][flash_id])+1
    if 1 not in _ids:
        continue
    _t_meas= np.array(data["times"][flash_id])
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

    #charge = []

    # we need to filter this so only the earliest time entry is kept

    all_bins += np.histogram2d(ids, t_meas, bins=(binids, offsets))[0]

if DEBUG:
    peaks = []
    import matplotlib.pyplot as plt
    from WCTECalib.utils import get_color 

print("... fitting hits")
for id in tqdm(range(len(all_bins))):
    this_wave = all_bins[id]
    def metric(params):
        sigma = 10**params[2]

        return np.sum((this_wave - params[0]*np.exp(-0.5*((offset_center - params[1])/sigma)**2))**2)

    x0 = (max(this_wave), offset_center[np.argmax(this_wave)], -1)
    bounds = [
                (0, np.inf),
                (-180, 180),
                (-5, 1)
            ]
    options={
        "eps":1e-5,
        "ftol":1e-20,
        "gtol":1e-20
    }
    res = minimize(metric, x0, bounds=bounds, options=options)
    cfd_time = -(10**res.x[2])*sqrt(-2*log(0.5)) + res.x[1]


    if DEBUG and id%50==0:
        plt.stairs(all_bins[id], offsets, color=get_color(id, 2014), alpha=1.)

    peaks.append(cfd_time)


        
plt.title("PMT {}".format(id))
plt.xlabel("Timing [ns]", size=14)
plt.savefig(os.path.join(os.path.dirname(__file__), "..","plotting","plots","raw_offset_distribution.png"), dpi=400)
plt.show()

ids = (0.5*(binids[1:] + binids[:-1])).astype(int)
    
positions = get_pmt_positions(ids)

distances = np.sqrt(np.sum( (positions - central_ball_loc)**2 , axis=1)) #predicted distances
pred_time = second*distances*N_WATER/C

use_offsets = np.array(peaks)-pred_time

new_df = deepcopy(df)

new_df["calc_offset"] = use_offsets[0]-use_offsets 

new_df.to_csv(
    os.path.join(os.path.dirname(__file__),"..", "data","calculated_offsets_lbmc.csv"),
    index=False
)
#np.array(df["unique_id"]), mean_offsets
