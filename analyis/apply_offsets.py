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
frequency = 200 #ns  - associated with a 5/3MHz frequency 

for ir, runno in tqdm(enumerate(range(len(pmt_hits)))):
    these_hit_pmts = pmt_hits[runno]
    these_charges = charges[runno]
    these_hit_times = times[runno] + true["offsets"][these_hit_pmts] + ir*frequency

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