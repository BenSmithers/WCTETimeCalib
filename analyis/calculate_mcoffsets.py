
import os 
import json 
import numpy as np 
from copy import deepcopy
from WCTECalib.utils import C, N_WATER, mm,ball_pos, second
from WCTECalib import df, N_CHAN, get_pmt_positions, N_MPMT
from tqdm import tqdm 
from scipy.signal import find_peaks
from scipy.optimize import minimize
from random import choice 

DEBUG =True

central_ball_loc = ball_pos

infile = os.path.join(
    os.path.dirname(__file__),
    "..",
    "data",
    "laserball_realdata_events.json"
)


offsets = np.linspace(-180*20, 180*20, 720*20)
offset_center = 0.5*(offsets[1:] + offsets[:-1])
binids = np.arange(-0.5, N_CHAN*N_MPMT+0.5, 1)+1 
all_bins = np.zeros(( len(binids)-1, len(offsets)-1))

#simulation result
_obj = open(infile, 'rt')
data = json.load(_obj)
_obj.close()

n_flash = len(data["times"])
reference_id = -1 
print("... Collecting hits")
print("Reference PMT - ".format(reference_id))
for flash_id  in tqdm(range(n_flash)):

    # ids in the data files are off by 1 relative to the geo file
    _ids = np.array(data["pmtid"][flash_id])+1
    if reference_id==-1:
        reference_id = _ids[0]
    if reference_id not in _ids:
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
    bin_where = choice(np.argwhere(ids==reference_id))


    _t_meas = _t_meas - t_meas[bin_where] 

    #charge = []

    # we need to filter this so only the earliest time entry is kept

    all_bins += np.histogram2d(_ids, _t_meas, bins=(binids, offsets))[0]

peaks = []
peak_width = []

if DEBUG:
    import matplotlib.pyplot as plt
    from WCTECalib.utils import get_color 

metrics = []
print("... fitting hits")
nplot = 0
maxplot = 20

for id in tqdm(range(len(all_bins))):
    this_wave = all_bins[id]/np.sum(all_bins[id])
    if np.sum(this_wave)==0:
        peaks.append(np.nan)
        peak_width.append(np.nan)
    def metric(params):
        sigma = 10**params[2]

        presum =(this_wave - params[0]*np.exp(-0.5*((offset_center - params[1])/sigma)**2))**2
        return np.sum(presum)

    x0 = (max(this_wave), offset_center[np.argmax(this_wave)], 1)
    bounds = [
                (0, np.inf),
                (-180, 180),
                (-5, 2)
            ]
    options={
        "eps":1e-5,
        "ftol":1e-20,
        "gtol":1e-20
    }
    res = minimize(metric, x0, bounds=bounds, options=options)
    #cfd_time = -(10**res.x[2])*sqrt(-2*log(0.5)) + res.x[1] 
    cfd_time = res.x[1] 
    goodness = metric(res.x)
    metrics.append(goodness)
    stepsize = offsets[1]-offsets[0]
    distance = 7./stepsize

    tpeak = find_peaks(this_wave, 0.4*np.max(this_wave), distance=distance)[0]

    
    shiftval = np.nansum(this_wave*offset_center)/np.nansum(this_wave)
    if nplot<maxplot and shiftval>1e-8:
        
        
        print(shiftval)
        plt.stairs(this_wave + nplot*0.025, offsets-shiftval, color=get_color(nplot/maxplot, 1, "inferno"), alpha=1., zorder=100-nplot, fill=True)
        plt.stairs(this_wave + nplot*0.025+0.001, offsets+0.5 - shiftval, color='k', alpha=0.3, zorder=100-nplot-0.5, fill=True)
        

        xfine = np.linspace(min(offset_center), max(offset_center), 3000)
        yfine = res.x[0]*np.exp(-0.5*((xfine - res.x[1])/(10**res.x[2]))**2)
        #plt.plot(xfine, yfine+ nplot*0.025, 'cyan', alpha=0.5, ls='--', zorder=100-nplot+0.25)
        nplot+=1
    peaks.append(cfd_time)
    peak_width.append(10**res.x[2])


plt.title("Various PMTs", size=14)
plt.xlabel("Earliest Relative Hit Time [ns]", size=14)
plt.ylabel("Arb. Units",size=14)
plt.savefig(os.path.join(os.path.dirname(__file__), "..","plotting","plots","raw_offset_distribution.png"), dpi=400)
plt.show()
plt.clf()

bins =np.linspace( np.nanmin(metrics), np.nanmax(metrics), 100)
binned_met = np.histogram(metrics, bins)
plt.stairs(binned_met[0], bins)
plt.xlabel("Fit Metric", size=14)
plt.ylabel("Counts",size=14)   
plt.show()

ids = (0.5*(binids[1:] + binids[:-1])).astype(int)
    
positions = get_pmt_positions(ids)

distances = np.sqrt(np.sum( (positions - central_ball_loc)**2 , axis=1)) #predicted distances
pred_time = second*distances*N_WATER/C

use_offsets = np.array(peaks)-pred_time

new_df = deepcopy(df)

new_df["calc_offset"] = use_offsets
new_df["offset_sigma"] = peak_width

new_df.to_csv(
    os.path.join(os.path.dirname(__file__),"..", "data","calculated_offsets_realdata.csv"),
    index=False
)
#np.array(df["unique_id"]), mean_offsets
