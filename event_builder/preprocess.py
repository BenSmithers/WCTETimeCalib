"""
1. Load trigger file, get all trigger times
2. Load a mPMT file. For each hit...
    a) Get the previous trigger time. Get triger number
    b) Subtract trigger time from hit time
    c) Apply offsets to get relative hit time
    d) Iterate over the hits, saving them to an events file
"""

from glob import glob 
import pandas as pd

import h5py as h5 
import numpy as np
import os 
import json 
from tqdm import tqdm 
import matplotlib.pyplot as plt 

offset_dict = pd.read_csv(
    os.path.join(os.path.dirname(__file__),
    "..",
    "data",
    "calculated_offsets_realdata.csv")
)
offset_id = offset_dict["unique_id"]
offset_time = offset_dict["calc_offset"]

with open(os.path.join(os.path.dirname(__file__),"..","WCTECalib","geodata",'PMT_Mapping.json'), 'r') as file:
    pmt_data = json.load(file)["mapping"]

def get_info(card_id, channel):
    """
        returns slot_id, pmt_pos
    """

    # parse them! 

    keys = (100*card_id)+channel 
    long_form_id = np.array([pmt_data[str(key)] for key in keys])

    return long_form_id//100, long_form_id%100 

class FileHandler:
    def __init__(self, root_folder =""):
        if root_folder=="":
            #self._root_folder = "/home/wcte/wcte/data/stand_alone/laser_ball/nominal_10min/processed/events"
            self._root_folder = "/home/wcte/wcte/data/stand_alone/beam_processed/events/"
        else:
            self._root_folder = root_folder
        self._active_file = None 
        self._active_bunch = -1 

        self.GROUPSIZE = 100000
    @property
    def root_folder(self):
        return self._root_folder
    
        
    def _activate_bunch(self, bunch, overwrite=False):
        if self._active_file is not None:
            self._active_file.close()
        
        filename = os.path.join(
            self._root_folder, 
            "beam_evt_part{}.hdf5".format(bunch)
        )
        if overwrite and os.path.exists(filename):
            os.remove(filename)
        self._active_file = h5.File(filename, 'a')
        self._active_bunch = bunch 
        return self._active_file

    def _flush_events(self, event, charge, time, ids):
        if "event{}".format(int(event)) in self._active_file.keys():
#            self._active_file["event{}/charge".format(int(event))].resize(
            if (self._active_file["event{}/time".format(int(event))].shape[0] + len(time))>2499:
                print("Max size exceeded - skipping. Evt ",event)
                return

            self._active_file["event{}/time".format(int(event))].resize((self._active_file["event{}/time".format(int(event))].shape[0] + len(time)),axis=0)
            self._active_file["event{}/time".format(int(event))][-len(time):] = time
            self._active_file["event{}/charge".format(int(event))].resize((self._active_file["event{}/charge".format(int(event))].shape[0] + len(charge)),axis=0)
            self._active_file["event{}/charge".format(int(event))][-len(charge):] = charge
            self._active_file["event{}/pmt_id".format(int(event))].resize((self._active_file["event{}/pmt_id".format(int(event))].shape[0] + len(ids)),axis=0)
            self._active_file["event{}/pmt_id".format(int(event))][-len(ids):] = ids 

        else:
            #self._active_file.create_dataset("event{}/charge".format(int(event) ), data=charge, maxshape=(500,))
            self._active_file.create_dataset("event{}/time".format(int(event)), data=time, maxshape=(2500,))
            self._active_file.create_dataset("event{}/pmt_id".format(int(event )), data=ids,maxshape=(2500,)) 
            self._active_file.create_dataset("event{}/charge".format(int(event )), data=ids,maxshape=(2500,)) 


    def write_events(self, eid, uid, charge, times):
        """
            Appends a series of event IDs to disc.
        """

        all_data = np.array([
            eid, uid, charge, times
        ]).T
        # sort the data according to the event ID 
        all_data = sorted(all_data, key=lambda x:x[0])
        all_data = np.transpose(all_data)
        # now, we save these in groups of 10,000 

        bunch_number = eid//self.GROUPSIZE
        for bunch in np.unique(bunch_number):
            self._activate_bunch(bunch)

            mask = bunch==bunch_number
            these_charges = all_data[2][mask]
            these_times = all_data[3][mask]
            these_pmts = all_data[1][mask]
            these_evts = all_data[0][mask]

            last_eid = -1
            charges = []
            times = []
            pmt_ids = []
            for hid in tqdm(range(len(these_evts))):
                if last_eid!=these_evts[hid] and last_eid!=-1:
                    self._flush_events(
                        last_eid, charges, times,  pmt_ids
                    )
                    charges =[]
                    times=[]
                    pmt_ids=[]
                charges.append( these_charges[hid] )
                times.append( these_times[hid] )
                pmt_ids.append(these_pmts[hid])
                last_eid = these_evts[hid]
            
            self._flush_events(
                last_eid, charges, times,  pmt_ids
            )
            
def process(filename, trigger_times):
    dfile = pd.read_parquet(filename)

    card_id = dfile["card_id"][0]
    if int(card_id)>130:
        return [], [], [], []
    
    channel = dfile["chan"]
    
    try:
        slot_id, pmt_pos = get_info(card_id, channel)
    except Exception:
        return [], [], [], []
#    slot_id, pmt_pos = get_info(np.array([card_id,]), np.array([channel,]))
    unique_id = slot_id*19 + pmt_pos
    
    if card_id not in [1,3,6,7, 8,11,12,14]:
        print("Skipping bad card {}".format(card_id))
        return [], [], [], []
    print("Processing Card {}".format(card_id))

    these_coarse    = dfile["coarse"]
    fine_time       = dfile["fine_time"]/65536.0
    charge = dfile["charge"]

    metadf = pd.DataFrame({"ID":unique_id.astype(int)})
    reference = pd.DataFrame({"ID":offset_id.astype(int), "time":offset_time})
    shifts = np.array(pd.merge(metadf, reference, on="ID", how="left")["time"], dtype=float)    
    total_time = (these_coarse + fine_time)*8 - shifts + 5000# add 5000 to push these _after_ the trigger signals

    trigger_number = np.digitize(total_time, trigger_times) - 1

    those_times = trigger_times[trigger_number]

    time_since_trigger = total_time - those_times

    #tmin = 130520
    #tmax = 130580

    tmin = 0 
    tmax = np.inf

    keep_mask = np.logical_and( trigger_number>=0, trigger_number<len(trigger_times)-1 )
    keep_mask = np.logical_and( keep_mask, np.logical_and( time_since_trigger>tmin, time_since_trigger<tmax ))
    return trigger_number[keep_mask], unique_id[keep_mask],charge[keep_mask],time_since_trigger[keep_mask]

def load_trigger(filename):
    dfile = pd.read_parquet(filename)
    card_id = dfile["card_id"][0]
    assert int(card_id)==131, "Trigger should be card 131"

    channel = 0
    data_mask = dfile["chan"] == channel
    
    these_coarse    = dfile["coarse"][data_mask]
    fine_time       = dfile["fine_time"][data_mask]/65536.0

    full_time = (these_coarse + fine_time)*8


    return np.array(sorted(full_time))


if __name__=="__main__":
    import sys 
    folder = sys.argv[1]

    print("Processing {}".format(folder))
    trigger_ip = "192.168.10.231"
    

    use_cfd = False 
    if use_cfd:
        all_files = glob(folder + "/*waveforms.parquet")
    else:
        all_files = glob(folder + "/*hits.parquet")
    print("Found {} files".format(len(all_files)))
    trigger_name = list(filter(lambda x:trigger_ip in x, all_files))
    assert len(trigger_name)==1, "There should only be one file matching the ip of the trigger"

    trigger_times = load_trigger(trigger_name)

    fh = FileHandler()

    binny = None # np.linspace(0, 1e6, 1000)
    histo = None # np.zeros(len(binny)-1)
    # Remove the trigger file 
    for filename in filter(lambda x:trigger_ip not in x, all_files):
        eid, uid, charge, time = process(filename, trigger_times) 
        if len(eid)==0:
            continue
        if binny is None:
            binny = np.linspace(min(time), 6e8, 2000)
            histo = np.zeros(len(binny)-1)
        
        histo += np.histogram(time, binny)[0]
        #fh.write_events(eid, uid, charge, time)

    plt.stairs(histo, binny)
    plt.savefig("./timing.png", dpi=400)