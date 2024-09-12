"""
Load MC file, apply previously sampeled offsets, save to disk
"""
import os 
import pandas as pd 
import numpy as np 
import json 

true = pd.read_csv(os.path.join(os.path.dirname(__file__), "..","data","offsets.csv"))

datafile = os.path.join(
    os.path.dirname(__file__),
    "..",
    "data",
    "wcsim.npz"
)

data = np.load(datafile, allow_pickle=True)

shift_times = []
shift_pmts = []
shift_charge = []

for runno in range(len(data["digi_hit_pmt"])):
    these_hit_pmts = data["digi_hit_pmt"][runno]
    these_charges = data["digi_hit_charge"][runno]
    these_hit_times = data["digi_hit_time"][runno]

    these_hit_times +=  true["offsets"][these_hit_pmts] 

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
    "wcsim_offset.json"
)
_obj = open(outfile, 'wt')
json.dump(save_data, _obj)
_obj.close()