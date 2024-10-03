"""
Load MC file, apply previously sampeled offsets

Then, smash all of hits into one long stream,
then, split the events out
"""
import os 
import pandas as pd 
import numpy as np 
import json 
from tqdm import tqdm 

from WCTECalib import N_MPMT, N_CHAN

true = pd.read_csv(os.path.join(os.path.dirname(__file__), "..","data","offsets.csv"))

datafile = os.path.join(
    os.path.dirname(__file__),
    "..",
    "data",
    "wcsim_highnoise.npz"
)

WINDOW_WIDTH = 200
HIT_THRESH=400


data = np.load(datafile, allow_pickle=True)

#preloading like this saves a TON of time on I/O
pmt_hits = data["digi_hit_pmt"][:]
charges = data["digi_hit_charge"][:]
times = data["digi_hit_time"][:]

shift_times = []
shift_pmts = []
shift_charge = []

frequency = 50e3  # Hz 
period = (1e9)/frequency

for ir, runno in tqdm(enumerate(range(len(pmt_hits)))):
    these_hit_pmts = pmt_hits[runno]
    these_charges = charges[runno]
    these_hit_times = np.array(times[runno] + true["offsets"][these_hit_pmts] + ir*period)
    
    shift_times+= these_hit_times.tolist()
    shift_pmts+=these_hit_pmts.tolist()
    shift_charge += these_charges.tolist()

if False:
    min_time = min(shift_times)
    max_time = max(shift_times)
    # should be about ~10 million hits over this whole timescale 
    print("... sampling noise hits")
    n_mean_extra_hits = int(N_CHAN*N_MPMT*(max_time-min_time)*(1e-9)*1e4)
    noisy_times = np.random.random(n_mean_extra_hits)*(max_time-min_time) + min_time
    pmt_ids = np.random.randint(0, N_MPMT*N_CHAN, n_mean_extra_hits)

    print(len(shift_times))
    shift_times+=noisy_times.tolist()
    print(len(shift_times))
    shift_pmts+=pmt_ids.tolist()
    shift_charge+=np.ones_like(pmt_ids).tolist()

all_data = np.array([
    shift_times, 
    shift_pmts, 
    shift_charge
]).T

sorted(all_data,key=lambda x:x[0])

hit_counter = 0
event_id = np.ones_like(shift_times)*-1

all_data = all_data.T
all_times = all_data[0]
hit_id = 0

levels = np.linspace(0, 1, 100)*len(all_times)
which_level = 0
while hit_id<len(all_times):
    if hit_id>levels[which_level]:
        which_level+=1 
        print("{}% done - hit {}".format(100*hit_id/len(all_times), hit_counter))

    now = all_times[hit_id]
    
    mask = np.logical_and( all_times>now-WINDOW_WIDTH, all_times<now+WINDOW_WIDTH)
    
    if np.sum(mask)>HIT_THRESH:
        if hit_counter!=event_id[hit_id]:
            # this is a new hit, step up the hit counter 
            hit_counter+=1 

        event_id[mask]=hit_counter

        hit_id = np.argwhere(mask)[-1][0]

    hit_id +=1
    
is_a_hit = event_id!=-1 

save_data = {
    "times":[], 
    "charge":[],
    "pmtid":[]
}


all_hitids = np.unique(event_id[is_a_hit])
for hit_id in all_hitids:
    mask = event_id==hit_id 
    save_data["times"].append(all_times[mask].tolist())
    save_data["charge"].append(all_data[2][mask].tolist())
    save_data["pmtid"].append(all_data[1][mask].tolist())


outfile = os.path.join(
    os.path.dirname(__file__),
    "..",
    "data",
    "wcsim_offset_recluster.json"
)
_obj = open(outfile, 'wt')
json.dump(save_data, _obj)
_obj.close()