import numpy as np 
import pandas as pd
from cfd.do_cfd import do_cfd, pmt_data, get_info
import matplotlib.pyplot as plt 
from scipy.stats import mode 
from scipy.optimize import minimize
from glob import glob 
from tqdm import tqdm
import os 
import h5py as h5 
from time import time as pytime

NS = 1.0 
COUNTERS = 8*NS
MAXNO = 1e6

offsets = pd.read_csv(os.path.join(os.path.dirname(__file__), "..","data","calculated_offsets_realdata.csv"))
offset_id = offsets["unique_id"]
offset_time = offsets["calc_offset"]


outfile = os.path.join(
    os.path.dirname(__file__),
    "..",
    "data",
    "laserball_realdata_events.h5"
)


def process_events(times, ids):
    """
        Assuming reverse order for now
    """
    THRESH = 75
    T_WIDTH = 10
    ENO = 1
    assert times[3]<times[2], "Wrong time order"
    assert len(times)==len(ids), "Irregular number of things!"

    index = 1
    max_index = 2

    eids = np.ones_like(ids)*-1 

    call_once = False 
    while (times[index]-times[max_index])<=T_WIDTH and max_index<len(times)-1:

        max_index+=1
        call_once = True 
    
    max_index -= 1
    
    
    n_incl = max_index - index + 1
    if n_incl>THRESH: # enough to trigger an event, so assign these to the next event 
        #print("Triggered with {} hits".format(n_incl))    
        new_max = times[max_index]+5 # an extra, 10 ns? 
        shift_max = max_index 
        call_once = False 
        while times[shift_max]<new_max and shift_max<len(times)-1:
            shift_max+=1
            call_once = True 
        assert call_once, "Didn't shift the shift over"
        eids[index:shift_max-1] = ENO

        index = shift_max+1
        max_index = index+1
        ENO +=1
    else:
        index += 1
        

    while max_index<len(times)-1:
        
        call_once = False 
        while (times[index]-times[max_index])<=T_WIDTH and max_index<len(times)-1:
            max_index+=1
            call_once = True  
        #assert call_once, "Didn't shift max_index over- {} and {}. {} and {}".format(index, max_index, times[index], times[max_index])
        max_index -= 1
        n_incl = max_index - index + 1

        if n_incl>THRESH: # enough to trigger an event, so assign these to the next event 
            print("Triggered with {} hits".format(n_incl))    
            new_min = times[max_index]-10 # an extra, eh 500 ns? 
            shift_max = max_index 
            call_once = False 
            while times[shift_max]>new_min and shift_max<len(times)-1:
                shift_max+=1
                call_once = True 
            #assert call_once, "Didn't shift the shift over"
            eids[index:shift_max-1] = ENO

            index = shift_max+1
            max_index = index+1
            ENO+=1
        else:
            index += 1
            if max_index<=index:
                max_index += 1
        
    print(ENO, "events")
    return eids

def reader(filename):
    
    card = []
    channel = []
    charge = []
    fine_time = []
    coarse = []

    all_files = glob(filename)
    all_files = sorted(all_files)
    
    for fn in all_files:
        print("Reading File {}".format(fn)) 
        data = pd.read_parquet(fn)
        these_cards = np.array(data["card_id"])
        these_chans = np.array(data["chan"])
        these_coarse = np.array(data["coarse"])
        keys = (100*these_cards)+these_chans
        mask = np.array([str(key) in pmt_data for key in keys])

        all_waves = data["samples"][mask]
        trimed_waves = []
        these_cards = these_cards[mask]
        these_chans = these_chans[mask]
        these_coarse = these_coarse[mask]
        
        print("Extracting Waveforms")
        sample_mask = []
        
        for i, wave in enumerate(all_waves):
            if these_cards[i]>129:
                sample_mask.append(False)
                continue

            if len(wave)==32:
                trimed_waves.append(wave)
                sample_mask.append(True)
            else:
                sample_mask.append(False)
        
        
        
        start = pytime()
        trimed_waves=-1*np.array(trimed_waves)
        times, amplitudes,baseline, add_filter = do_cfd(trimed_waves)
        end = pytime()
        print(end - start, "Seconds")

        card += these_cards[sample_mask][add_filter].tolist()
        channel += these_chans[sample_mask][add_filter].tolist()
        charge += amplitudes.tolist()
        fine_time += times.tolist()
        coarse+= these_coarse[sample_mask][add_filter].tolist()

    channel= np.array(channel).flatten()
    card = np.array(card).flatten()
    
    fine_time = np.array(fine_time).flatten()
    
    

    coarse = np.array(coarse).flatten()
    
    charge = np.array(charge).flatten()
    time =  ((coarse + fine_time)*COUNTERS) #  % PERIOD
    slot_id, pmt_pos = get_info(card, channel)
    
    #all_shifties =np.array([ offset_time[ offset_id == (slot_id*19+pmt_pos)[i] ] for i in range(len(fine_time))])

    metadf = pd.DataFrame({"ID": (19*slot_id+pmt_pos).astype(int)})
    reference = pd.DataFrame({"ID":offset_id.astype(int), "time":offset_time})
    all_shifties = np.array(pd.merge(metadf, reference, on="ID", how="left")["time"], dtype=float)    
    add_mask = np.logical_and(np.array([sid not in [57, 58, 81] for sid in slot_id]).flatten(), np.logical_not(np.isnan(all_shifties)))
    all_data = np.array([
        time[add_mask]-all_shifties[add_mask], charge[add_mask], slot_id[add_mask], pmt_pos[add_mask]
    ]).T

    all_data = sorted(all_data, key=lambda x:x[0], reverse=True)
    all_data = np.transpose(all_data)
    print(len(all_data[0]), "hits")
    print("{}Order".format("Reverse " if all_data[0][0]>all_data[0][-1] else "Normal "))
    event = process_events(all_data[0], all_data[2])

    pmt_id = 19*all_data[2] + all_data[3] 
    

    

    dfile = h5.File(outfile, 'w')
    charges = []
    times = []
    pmt_ids = []

    last_eid = -1 
    for hid in tqdm(range(len(all_data[0]))):
        if last_eid!=event[hid] and last_eid!=-1:
            dfile.create_dataset("event{}/charge".format(int(last_eid) ), data=charges)
            dfile.create_dataset("event{}/time".format(int(last_eid)), data=np.array(times))
            dfile.create_dataset("event{}/pmt_id".format(int(last_eid )), data=pmt_ids) 
            charges =[]
            times=[]
            pmt_ids=[]
        if pmt_id[hid] in pmt_ids:
            continue
        charges.append(all_data[1][hid])
        times.append(all_data[0][hid])
        pmt_ids.append(pmt_id[hid])

        last_eid = event[hid]

    # once more to clear out what's left
    dfile.create_dataset("event{}/charge".format(int(event[hid]) ), data=charges)
    dfile.create_dataset("event{}/time".format(int(event[hid] )), data=np.array(times))
    dfile.create_dataset("event{}/pmt_id".format(int(event[hid] )), data=pmt_ids) 

    dfile.close()    

if __name__=="__main__":
    import sys 
    reader(sys.argv[1])