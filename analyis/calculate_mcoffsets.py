
import os 
import json 
import numpy as np 
from copy import deepcopy
from WCTECalib.utils import C, N_WATER, mm,ball_pos, second, set_axes_equal
from WCTECalib import df, N_CHAN, get_pmt_positions, N_MPMT
from tqdm import tqdm 
from scipy.signal import find_peaks
from scipy.optimize import minimize
from scipy.stats import mode 
from random import choice 
import h5py as h5 
from math import nan
DEBUG =True
force_calc = True 

central_ball_loc = ball_pos

infile = os.path.join(
    os.path.dirname(__file__),
    "..",
    "data",
    "laserball_realdata_events.h5"
)

binned_data_file = os.path.join(
    os.path.dirname(__file__),
    ".processed_data.csv"
)

offsets = np.linspace(-180*7, 180*7, 720*10)
offset_center = 0.5*(offsets[1:] + offsets[:-1])
binids = np.arange(-0.5, N_CHAN*N_MPMT+0.5, 1) 

just_id_bins = np.zeros(N_CHAN*N_MPMT+1)
if os.path.exists(binned_data_file) and not force_calc:
    print("LOADING PRE-SAVED BINNED DATA")
    all_bins = np.loadtxt(binned_data_file) 
    reference_id = 112
else:

    all_bins = np.zeros(( len(binids)-1, len(offsets)-1))

    #simulation result
    data = h5.File(infile, 'r')

    # this way we do the I/O all at once; this is significantly faster 

    reference_id = 112
    n_flash = len(data.keys())

    print("... {} Events".format(n_flash))
    print("... Collecting hits")
    print("Reference PMT - {}".format(reference_id))
    if False:
        for _flash_id in tqdm(range(n_flash)):
            flash_id = _flash_id+1
            for id in np.array(data["event{}/pmt_id".format(flash_id)]):
                just_id_bins[int(id)] +=1
                
        reference_id = np.argmax(just_id_bins)
    print("Set reference ID to {}".format(reference_id))


    for _flash_id  in tqdm(range(2000)):
        if _flash_id==0:
            print("Skipping fake event")
            continue
        flash_id = _flash_id+1



        

        # ids in the data files are off by 1 relative to the geo file
        if not "event{}/pmt_id".format(flash_id) in data:
            continue
        _ids = np.array(data["event{}/pmt_id".format(flash_id)]) 
        if reference_id not in _ids:
            continue
        _t_meas= np.array(data["event{}/time".format(flash_id)])
        _charge = np.array(data["event{}/charge".format(flash_id)])

        these_data = np.array([_ids, _t_meas, _charge]).T 
        these_data = np.array(sorted(these_data, key=lambda x:x[1] , reverse=True))

        ids = []
        charges = []
        t_meas = []
        for entry in these_data:
            if entry[0] not in ids:
                ids.append(entry[0])
                charges.append(entry[2])
                t_meas.append(entry[1])

        
        ids = np.array(ids)
        bin_where = np.argwhere(ids==reference_id)[0]
        t_meas = np.array(t_meas) - np.array(_t_meas[bin_where])


        #charge = []

        # we need to filter this so only the earliest time entry is kept

        all_bins += np.histogram2d(ids, t_meas, bins=(binids, offsets))[0]



    np.savetxt(binned_data_file, all_bins)

if DEBUG:
    import matplotlib.pyplot as plt
    from WCTECalib.utils import get_color 
peaks = []
heights = []
peak_width = []
metrics = []
print("... fitting hits")
nplot = 0
maxplot = 20
n_good = 0
n_bad = 0 
not_there = 0
shift_time = -1

is_good = []

for id in tqdm(range(len(all_bins))):
    this_wave = all_bins[id] #np.sum(all_bins[id])
    renorm = 1.0
    #renorm = 1/np.sum(all_bins[id])
    if np.sum(this_wave)==0:
        peaks.append(np.nan)
        peak_width.append(np.nan)
        is_good.append(0)
        not_there += 1
        continue
    def metric(params):
        sigma =params[2]

        presum =(this_wave - params[0]*np.exp(-0.5*((offset_center - params[1])/sigma)**2))**2
        #vlate = this_wave- params[1] > 2
        #presum[vlate]= 0.0 

        return np.nansum(presum)
    
    x0 = (max(this_wave), offset_center[np.argmax(this_wave)], 2)
    bounds = [
                (0, np.inf),
                (-1000, 1000),
                (0, 20)
            ]
    options={
        "eps":1e-5,
        "ftol":1e-10,
        "gtol":1e-10
    }
    res = minimize(metric, x0, bounds=bounds, options=options)
    #cfd_time = -(10**res.x[2])*sqrt(-2*log(0.5)) + res.x[1] 
    cfd_time = res.x[1] 
    goodness = metric(res.x)
    metrics.append(goodness)
    stepsize = offsets[1]-offsets[0]
    distance = 7./stepsize

    tpeak = find_peaks(this_wave, 0.6*np.nanmax(this_wave), distance=distance)[0]

    if len(tpeak)!=1:#  or np.log10(goodness)<1.5:
        print("Two peaks")
        shiftval = np.nan
        n_bad +=1 
        is_good.append(0)
        peaks.append(np.nan)
        peak_width.append(np.nan)

        continue
    else:
        n_good +=1 
        is_good.append(1)
        shiftval = cfd_time # offset_center[tpeak[0]] # cfd_time
    
    if  nplot<maxplot and (np.log10(goodness)>1.5) and (not np.isnan(shiftval)):
        print(shiftval)
        plt.stairs(this_wave*renorm + nplot*0.025, offsets-shiftval*0.59, color=get_color(nplot/maxplot, 1, "inferno"), alpha=1., zorder=100-nplot, fill=True)
        plt.stairs(this_wave*renorm + nplot*0.025+0.001, offsets+0.5 - shiftval*0.59, color='k', alpha=0.3, zorder=100-nplot-0.5, fill=True)
        

        xfine = np.linspace(min(offset_center), max(offset_center), 3000)
        yfine = res.x[0]*np.exp(-0.5*((xfine - res.x[1])/(res.x[2]))**2)
        plt.plot(xfine-shiftval*0.59, yfine*renorm+ nplot*0.025, 'cyan', alpha=0.5, ls='--', zorder=100-nplot+0.25)
        nplot+=1
    peaks.append(shiftval)
    peak_width.append(res.x[2])
    heights.append(res.x[0])

print("{} good, {} bad, {} gone".format(n_good, n_bad, not_there))

plt.title("Various PMTs", size=14)
plt.xlabel("Earliest Relative Hit Time [ns]", size=14)
plt.ylabel("Arb. Units",size=14)
plt.xlim([-20, 20])
plt.savefig(os.path.join(os.path.dirname(__file__), "..","plotting","plots","raw_offset_distribution.png"), dpi=400)
plt.show()
plt.clf()

bins =np.linspace( -1, 6, 1000)
binned_met = np.histogram(np.log10(metrics), bins)
plt.stairs(binned_met[0], bins)
plt.xlabel("Fit Metric", size=14)
plt.ylabel("Counts",size=14)   
plt.show()


ids = (0.5*(binids[1:] + binids[:-1])).astype(int)
    




positions = []
mask = []
for uid in ids:
    pos = get_pmt_positions([uid,])
    if uid in [1899, 1885, 1896, 1898]:
        print("Found it...")
        print(uid)
        print(pos)

    if len(pos)!=0:
        positions.append(pos[0])
        mask.append(True)
    else:
        mask.append(False)
mask = np.array(mask).flatten()
print(np.shape(positions))
positions = np.array(positions)
is_good = np.array(is_good)[mask]


ax = plt.axes(projection="3d")
scatty = ax.scatter(positions.T[0], positions.T[1], positions.T[2], c=is_good.astype(int),vmin=0, vmax=1, cmap=plt.cm.inferno)
#ax.plot(xs, ys, zs, 'bo')
ax.set_xlabel("X [m]")
ax.set_ylabel("Y [m]")
ax.set_zlabel("Z [m]")
set_axes_equal(ax)
plt.show()




distances = np.sqrt(np.sum( (positions - central_ball_loc)**2 , axis=1)) #predicted distances
pred_time = second*distances*N_WATER/C
use_offsets = np.array(peaks)[mask]-pred_time

new_df = deepcopy(df)
new_df["calc_offset"] = use_offsets  
new_df["offset_sigma"] = np.array(peak_width)[mask]

#new_df["nhits"] = np.array(heights)[mask]

if True:
    new_df.to_csv(
        os.path.join(os.path.dirname(__file__),"..", "data","calculated_offsets_realdata.csv"),
        index=False
    )
#np.array(df["unique_id"]), mean_offsets
