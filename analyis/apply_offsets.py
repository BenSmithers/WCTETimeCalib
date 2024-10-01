"""
Load MC file, apply previously sampeled offsets, save to disk
"""
import os 
import pandas as pd 
import numpy as np 
import json 
from tqdm import tqdm 

true = pd.read_csv(os.path.join(os.path.dirname(__file__), "..","data","offsets.csv"))

datafile = os.path.join(
    os.path.dirname(__file__),
    "..",
    "data",
    "wcsim.npz"
)

data = np.load(datafile, allow_pickle=True)

#preloading like this saves a TON of time on I/O
pmt_hits = data["digi_hit_pmt"][:]
charges = data["digi_hit_charge"][:]
times = data["digi_hit_time"][:]

shift_times = []
shift_pmts = []
shift_charge = []

phase_lock = False 
frequency = 200000 #ns  - associated with a 5MHz frequency 

# choose a five bad PMTs 
bad_ids = np.array(range(1843))
bad_ids = bad_ids[bad_ids%300 == 1]
print(bad_ids)
# choose an index where the clock changes 
flipflop = np.random.randint(50, len(pmt_hits),5)


for ir, runno in tqdm(enumerate(range(len(pmt_hits)))):
    these_hit_pmts = pmt_hits[runno]
    these_charges = charges[runno]
    these_hit_times = np.array(times[runno] + true["offsets"][these_hit_pmts] + ir*frequency)
    
    if False:
        for j, flop in enumerate(flipflop):
                
            if ir>flop:
                mask = np.where(np.in1d(these_hit_pmts, bad_ids[j]))[0]
                if np.sum(mask.astype(int))!=0:
                    these_hit_times[mask] = these_hit_times[mask] + 8

    shift_times.append(these_hit_times.tolist())
    shift_pmts.append(these_hit_pmts.tolist())
    shift_charge.append(these_charges.tolist())


save_data = {
    "times":shift_times, 
    "charge":shift_charge,
    "pmtid":shift_pmts
}

outfile = os.path.join(
    os.path.dirname(__file__),
    "..",
    "data",
    "wcsim_offset_lock.json"
)
_obj = open(outfile, 'wt')
json.dump(save_data, _obj)
_obj.close()