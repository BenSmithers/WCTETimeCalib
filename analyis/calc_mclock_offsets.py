import matplotlib.pyplot as plt 
import os 
import json 
from math import log, sqrt 
import numpy as np 
from copy import deepcopy
from WCTECalib.utils import C, N_WATER, mm,ball_pos, second, get_color
from WCTECalib import df, N_CHAN, get_pmt_positions, N_MPMT
from scipy.optimize import minimize
from tqdm import tqdm
from scipy.signal import find_peaks

pmt_no = 0
infile = os.path.join(
    os.path.dirname(__file__),
    "..",
    "data",
    "wcsim_offset_swing.json"
)

PHASELOCK = 200000.0# 0.005
outname = os.path.join(os.path.dirname(__file__),"..", "data","calculated_offsets_lock_swing.csv")

print(PHASELOCK)
maxtime = min([600, PHASELOCK])
offsets = np.linspace(0, maxtime, 1201)
offset_center = 0.5*(offsets[1:] + offsets[:-1])
# we need to offset the IDs by one to match WCSim
binids = np.arange(-0.5, N_CHAN*N_MPMT+0.5, 1) +1
ids = np.array(range(N_CHAN*N_MPMT)) +1
print(N_CHAN*N_MPMT)

all_bins = np.zeros(( len(binids)-1, len(offsets)-1))

#simulation result
_obj = open(infile, 'rt')
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

    _t_meas= (np.array(hit_times[flash_id])+0.5*maxtime)  -flash_id*PHASELOCK
    _charge = np.array(hit_charges[flash_id])

    all_bins += np.histogram2d(_ids, _t_meas, bins=(binids, offsets))[0]


all_times = []
peak_widths=[]
metrics = []
num_bad = 0
called_once = False
print("... fitting distributions")
for i in tqdm(range(len(all_bins))):

    this_wave = all_bins[i]/np.sum(all_bins[i])
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
    #cfd_time = -(10**res.x[2])*sqrt(-2*log(0.5)) + res.x[1]
    cfd_time = res.x[1] 
    all_times.append(cfd_time)
    this_met = metric(res.x)
    peak_widths.append(10**res.x[2])
    metrics.append(this_met)

    stepsize = offsets[1]-offsets[0]
    distance = 7./stepsize

    tpeak = find_peaks(this_wave, 0.4*np.max(this_wave), distance=distance)[0]
    if  len(tpeak)>1:
        called_once = True
        xfine = np.linspace(0, PHASELOCK, 3000)
        yfine = res.x[0]*np.exp(-0.5*((xfine - res.x[1])/(10**res.x[2]))**2)
        #print(res.x[2])
        #print("One at {}".format(res.x[1]))
        #plt.vlines(res.x[1], 0,40, color='k', alpha=0.1,zorder=0)
        #plt.plot(xfine, yfine+ num_bad*0.025, 'cyan', alpha=0.5, ls='--', zorder=100-num_bad+0.25)
        #plt.vlines(cfd_time, [0,], [max(yfine),], color='k', zorder=0)
        
        plt.stairs(this_wave + num_bad*0.025+0.001, offsets+0.5, color='k', alpha=0.3, zorder=100-num_bad-0.5, fill=True)
        plt.stairs(this_wave + num_bad*0.025, offsets, color=get_color(i*0.25 + num_bad*50, 850, "inferno"), alpha=1., zorder=100-num_bad, fill=True)
        num_bad +=1
if called_once:
    plt.xlabel(r"Time mod $f$ [ns]", size=14)
    #plt.xlim([10, 120])
    plt.ylabel("Arb. Units",size=14)
    plt.ylim([0, 0.55])
    plt.savefig("../plotting/plots/mod_time_distribution.png", dpi=400)
    
    plt.show()
    

# metric! 
plt.clf()
qbin = np.linspace(min(metrics), max(metrics), 200)
plt.hist(metrics, bins=qbin)
plt.yscale('log')
plt.xlabel("Fit Metric",size=14)
plt.ylabel("Arb. Units", size=14)
plt.show()



all_times = np.array(all_times) -pred_time

new_df = deepcopy(df)

new_df["calc_offset"] = all_times[0] - all_times
new_df["offset_sigma"] = peak_widths

new_df.to_csv(
    outname,
    index=False
)