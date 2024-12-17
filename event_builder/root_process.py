import uproot
import numpy as np
import h5py as h5 
from tqdm import tqdm 
import json
import os 

def parse(filename):
    data = uproot.open(filename)["WCTEReadoutWindows;1"]

    # all of the things 
    print("Extracting the things")
    charge = data["hit_pmt_charges"].array(library="np")
    times = data["hit_pmt_times"].array(library="np")
    slot = data["hit_mpmt_slot_ids"].array(library="np")
    channel = data["hit_pmt_channel_ids"].array(library="np")

    save_data = {
        "times":[], 
        "charge":[],
        "pmtid":[]
    }

    for eid in tqdm(range(len(charge))):
        if len(times[eid])==0:
            continue
        
        save_data["pmtid"].append((slot[eid]*19 + channel[eid]).tolist())
        save_data["charge"].append( charge[eid].tolist() )
        save_data["times"].append((times[eid] - np.min(times[eid])).tolist())

            
        
    outfile = os.path.join(
        os.path.dirname(__file__),
        "..",
        "data",
        "laserball_realdata_events.json"
    )

    _obj = open(outfile, 'wt')
    json.dump(save_data, _obj)
    _obj.close()

if __name__=="__main__":
    import sys 
    if len(sys.argv)<2:
        print("Give filename...")
    else:
        parse(sys.argv[1])