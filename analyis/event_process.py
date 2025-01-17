
import pandas as pd 
import os 
import numpy as np 
from tqdm import tqdm 
import h5py as h5 
from WCTECalib.utils import  get_color

import matplotlib.pyplot as plt 


offset_dict = pd.read_csv(
    os.path.join(os.path.dirname(__file__),
    "..",
    "data",
    "calculated_offsets_realdata.csv")
)
unique_ids = np.array(offset_dict["unique_id"])
is_bad = np.isnan(offset_dict["calc_offset"])

bad_ids = []
for i, uid in enumerate(unique_ids):
    if is_bad[i]:
        bad_ids.append(uid)
bad_ids = np.array(bad_ids)
def process(event):


    _ids = np.array(event["pmt_id"])
    _t_meas= np.array(event["time"])
    _charge = np.array(event["charge"])

    good_mask = np.logical_not([i in bad_ids for i in _ids])

    ids = _ids[good_mask].astype(int)
    t_meas = _t_meas[good_mask]
    charge = _charge[good_mask]


    # okay now get the offsets

    these_offs = np.array([offset_dict.loc[offset_dict["unique_id"] == entry, "calc_offset" ] for entry in ids]).flatten()    

    # trim the late-hits 
    early_id = []
    early_time = []
    early_charge = []
    for i in range(len(ids)):
        hit_id = ids[i]
        if hit_id in early_id:
            continue
        else:
            early_id.append(hit_id)
            early_time.append(t_meas[i] - these_offs[i])
            early_charge.append(charge[i])


    # we want to mask out 
    return np.array(early_id), np.array(early_time), np.array(early_charge)


if __name__=="__main__":
    infile = os.path.join(
        os.path.dirname(__file__),
        "..",
        "data",
        "laserball_realdata_events.h5"
    )

    data = h5.File(infile)

    n_flash = len(data.keys())


    for _flash_id in tqdm(range(n_flash)):
        flash_id = _flash_id+1
        ids, time, charge = process(data["event{}".format(flash_id)])  
        
        
        bins = np.linspace(np.nanmean(time)-30, np.nanmean(time)+30)
    

        mpmt_no = ids // 19
        channel = ids % 19

        stacked = np.zeros(len(bins)-1)
        zthis = 100

        for mpmt in np.unique(mpmt_no):
            binned = np.histogram((time)[mpmt_no==mpmt], bins, )[0] 
            plt.stairs(binned+stacked, bins, zorder= zthis, color=get_color(zthis, 100, "jet"), fill=True)
            zthis -= 1
            stacked += binned
            if zthis<0:
                break

        plt.xlabel("Times")
        plt.show()