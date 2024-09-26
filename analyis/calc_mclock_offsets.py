import matplotlib.pyplot as plt 
import os 
import json 
from math import log, sqrt 
import numpy as np 
from copy import deepcopy
from WCTECalib.utils import C, N_WATER, mm,ball_pos, second, get_color
from WCTECalib.alt_geo import df, N_CHAN, get_pmt_positions, N_MPMT
from scipy.optimize import minimize
from tqdm import tqdm

pmt_no = 0

outfile = os.path.join(
    os.path.dirname(__file__),
    "..",
    "data",
    "wcsim_offset_lock.json"
)

PHASELOCK = 200
offsets = np.linspace(0, PHASELOCK, 2401)
offset_center = 0.5*(offsets[1:] + offsets[:-1])
# we need to offset the IDs by one to match WCSim
binids = np.arange(-0.5, N_CHAN*N_MPMT+0.5, 1) +1
ids = np.array(range(N_CHAN*N_MPMT)) +1
print(N_CHAN*N_MPMT)

all_bins = np.zeros(( len(binids)-1, len(offsets)-1))

#simulation result
_obj = open(outfile, 'rt')
data = json.load(_obj)
_obj.close()

hit_ids = list(data["pmtid"][:])
hit_charges = list(data["charge"][:])
hit_times = list(data["times"][:])

n_flash = len(data["times"])

positions = get_pmt_positions(ids)

distances = np.sqrt(np.sum( (positions - ball_pos)**2 , axis=1)) #predicted distances
pred_time = second*distances*N_WATER/C

print("... binning data")
for flash_id  in tqdm(range(n_flash)):

    
    _ids = np.array(hit_ids[flash_id])+1

    _t_meas= (np.array(hit_times[flash_id])+0.5*PHASELOCK) -flash_id*PHASELOCK
    _charge = np.array(hit_charges[flash_id])

    all_bins += np.histogram2d(_ids, _t_meas, bins=(binids, offsets))[0]


all_times = []
called_once = False
print("... fitting distributions")
for i in tqdm(range(len(all_bins))):

    this_wave = all_bins[i]
    def metric(params):
        sigma = 10**params[2]

        return np.sum((this_wave - params[0]*np.exp(-0.5*((offset_center - params[1])/sigma)**2))**2)

    x0 = (max(this_wave), offset_center[np.argmax(this_wave)], -1)
    bounds = [
                (0, np.inf),
                (0, PHASELOCK),
                (-5, 1)
            ]
    options={
        "eps":1e-5,
        "ftol":1e-20,
        "gtol":1e-20
    }
    res = minimize(metric, x0, bounds=bounds, options=options)
    cfd_time = -(10**res.x[2])*sqrt(-2*log(0.5)) + res.x[1]
    all_times.append(cfd_time)

    if (10**res.x[2])>1.0:
        called_once = True
        xfine = np.linspace(0, PHASELOCK, 3000)
        yfine = res.x[0]*np.exp(-0.5*((xfine - res.x[1])/(10**res.x[2]))**2)
        #print(res.x[2])
        #print("One at {}".format(res.x[1]))
        #plt.vlines(res.x[1], 0,40, color='k', alpha=0.1,zorder=0)
        plt.plot(xfine, yfine, 'red', alpha=0.2, zorder=11)
        plt.vlines(cfd_time, [0,], [max(yfine),], color='k', zorder=0)
        plt.stairs(all_bins[i], offsets, color=get_color(i+20, 2033), zorder=10)

if called_once:
    plt.xlabel("Time [ns]", size=14)
    plt.savefig("../plotting/plots/mod_time_distribution.png", dpi=400)
    plt.show()

all_times = np.array(all_times) #- pred_time

new_df = deepcopy(df)

new_df["calc_offset"] = all_times[0] - all_times

new_df.to_csv(
    os.path.join(os.path.dirname(__file__),"..", "data","calculated_offsets_lock.csv"),
    index=False
)