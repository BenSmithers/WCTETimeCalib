import pandas as pd 
import numpy as np
import sys 
import os 
import json 
from glob import glob 
from cfd.do_cfd import do_cfd
import matplotlib.pyplot as plt
NS = 1.0 
COUNTERS = 8*NS

# Open and read the JSON file
with open(os.path.join(os.path.dirname(__file__), "..","WCTECalib","geodata",'PMT_Mapping.json'), 'r') as file:
    pmt_data = json.load(file)["mapping"]

def get_info(card_id, channel):
    """
        returns slot_id, pmt_pos
    """

    # parse them! 

    keys = (100*card_id)+channel 
    long_form_id = np.array([pmt_data[str(key)] for key in keys])

    return long_form_id//100, long_form_id%100 

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
        
        print("Extracting Waveforms")
        waveforms = []
        oldmask = []
        for i, wave in enumerate(data["samples"]):

            if len(wave)==32:
                waveforms.append(wave)
                #plt.bar(range(32),wave, alpha=0.1, color='k') 
                oldmask.append(True)
            else:
                oldmask.append(False)
        waveforms=-1*np.array(waveforms)
        times, amplitudes,baseline , good_mask= do_cfd(waveforms)
        fine_time += times.tolist()
        charge += amplitudes.tolist()

        card += np.array(data["card_id"][oldmask][good_mask]).tolist()
        channel += np.array(data["chan"][oldmask][good_mask]).tolist()
        #charge += np.array(data["charge"][:]).tolist()
        coarse += np.array(data["coarse"][oldmask][good_mask]).tolist()
        break


    card = np.array(card)
    channel= np.array(channel)


    keys = (100*card)+channel 

    mask = np.array([str(key) in pmt_data for key in keys])

    card = card[mask]
    channel = channel[mask]
    
    
    fine_time = np.array(fine_time).flatten()[mask]
    coarse = np.array(coarse).flatten()[mask]

    slot_id, pmt_pos = get_info(card, channel)
    charge = np.array(charge).flatten()[mask]

    time =  (coarse + fine_time)
    
    mask = np.logical_not(np.isnan(charge))

    
    all_data = np.array([
        time[mask], charge[mask], slot_id[mask], pmt_pos[mask]    
    ]).T 

    all_data = sorted(all_data, key=lambda x:x[0], reverse=False)

    print("Saving file")
    np.savetxt("hits.csv", all_data, delimiter=", ")



if __name__=="__main__":
    print(sys.argv[1])
    data = reader(sys.argv[1])