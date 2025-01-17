import numpy as np 
import json 
import os 
import h5py as h5 
from tqdm import tqdm

target = os.path.join(os.path.dirname(__file__), "events.csv")

print("Loading in data")
data = np.loadtxt(target, delimiter=",", skiprows=1).transpose()

hit_id = data[0][:]
all_times = data[1][:]
charge = data[2][:]
slot= data[3][:]
channel = data[4][:]
event = data[5]

pmt_id = 19*slot + channel 

save_data = {
        "times":[], 
        "charge":[],
        "pmtid":[]
    }

outfile = os.path.join(
    os.path.dirname(__file__),
    "..",
    "data",
    "laserball_realdata_events.h5"
)


dfile = h5.File(outfile, 'w')
charges = []
times = []
pmt_ids = []

last_eid = -1 
for hid in tqdm(range(len(hit_id))):
    if last_eid!=event[hid] and last_eid!=-1:
        dfile.create_dataset("event{}/charge".format(int(last_eid) ), data=charges)
        dfile.create_dataset("event{}/time".format(int(last_eid)), data=np.array(times))
        dfile.create_dataset("event{}/pmt_id".format(int(last_eid )), data=pmt_ids) 
        charges =[]
        times=[]
        pmt_ids=[]

    charges.append(charge[hid])
    times.append(all_times[hid])
    pmt_ids.append(pmt_id[hid])

    last_eid = event[hid]
# once more to clear out what's left
dfile.create_dataset("event{}/charge".format(int(event[hid]) ), data=charges)
dfile.create_dataset("event{}/time".format(int(event[hid] )), data=np.array(times))
dfile.create_dataset("event{}/pmt_id".format(int(event[hid] )), data=pmt_ids) 

dfile.close()
