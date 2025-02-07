import numpy as np 
import pandas as pd
from cfd.do_cfd import do_cfd, pmt_data, get_info
import matplotlib.pyplot as plt 
from scipy.stats import mode 
from scipy.optimize import minimize
from WCTECalib import df, get_pmt_positions
from WCTECalib.utils import ball_pos, second, C, N_WATER
import os 
from copy import deepcopy 
from tqdm import tqdm
import h5py as h5 

filename = "../data/laserball/laser_ball_20241205183135_6_waveforms.parquet"

PERIOD = 262144
dfile = pd.read_parquet(filename)


outfile = os.path.join(
    os.path.dirname(__file__),
    "..",
    "data",
    "laserball_realdata_events.h5"
)



# ----------- Grab only valid hits on PMTs
chan = np.array(dfile["chan"])
mPMT_card = np.array(dfile["card_id"])
keys=  (100*mPMT_card) + chan
valid = np.array([str(key) in pmt_data for key in keys])

# ----------- Apply that validity cut 
slot_id, pmt_pos = get_info(mPMT_card[valid], chan[valid])
valid_waves = dfile["samples"][valid]
valid_counters = dfile["coarse"][valid]
chan = chan[valid]
mPMT_card = mPMT_card[valid]

# ----------- Now find the waveforms with the correct number of samples
selected = []
keep_mask = [] 
for i, wave in enumerate(valid_waves):
    if len(wave) == 32: # other lengths have problems 
        selected.append(wave)
        keep_mask.append(True)
    else:
        keep_mask.append(False)
selected = -1*np.array(selected)

# ---------- Run the CFD and cut any CFD failures 
print("Running CFD")
fine_time, amp, base, cfd_filter = do_cfd(selected)    
coarse_cut =np.array( valid_counters[keep_mask][cfd_filter])

# ---------- Apply those last two cuts. Apply modulus to get times and events
event_time = (( coarse_cut+ fine_time)*8) % PERIOD

event_coarse = (coarse_cut*8) % PERIOD # we should be able to get a rough idea of when the pulses are based on the coarse counter with the most hits 
t_time = mode(event_coarse).mode

good = event_coarse > (t_time - 500)

event_number= (( coarse_cut+ fine_time)*8) // PERIOD
slots = slot_id[keep_mask][cfd_filter]
channels = chan[keep_mask][cfd_filter]
pmt_id = 19*slots + channels

print(len(event_time), len(event_number), len(slots), len(pmt_id), len(amp))

flash_numbers = np.unique(event_number)

print("Exporting Events")
dfile = h5.File(outfile, 'w')
eno = 0
for fid in tqdm(flash_numbers):
    evt_mask = np.logical_and(event_number==fid, good)
    these_times =event_time[evt_mask]
    these_ids = pmt_id[evt_mask]
    
    these_mpmts = slots[evt_mask]
    
    hit_ids = np.unique(these_mpmts)
    if len(hit_ids)<10:
        continue
    else:
        eno += 1
        
        
        dfile.create_dataset("event{}/pmt_id".format(int(eno) ), data=these_ids)
        dfile.create_dataset("event{}/time".format(int(eno) ), data=these_times)
        dfile.create_dataset("event{}/charge".format(int(eno) ), data=amp[evt_mask])
    
dfile.close()