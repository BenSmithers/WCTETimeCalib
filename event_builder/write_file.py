import pandas as pd 
import numpy as np
import sys 
import os 
import json 
from glob import glob 
from cfd.do_cfd import do_cfd, get_info, pmt_data
import matplotlib.pyplot as plt
NS = 1.0 
COUNTERS = 8*NS
PERIOD = 262144

# Open and read the JSON file


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
            if these_cards[i]==131:
                print("That's a trigger thing")
            if len(wave)==32:
                trimed_waves.append(wave)
                #plt.bar(range(32),wave, alpha=0.1, color='k') 
                sample_mask.append(True)
            else:
                sample_mask.append(False)
        
        
        trimed_waves=-1*np.array(trimed_waves)
        times, amplitudes,baseline, add_filter = do_cfd(trimed_waves)

        card += these_cards[sample_mask][add_filter].tolist()
        channel += these_chans[sample_mask][add_filter].tolist()
        charge += amplitudes.tolist()
        fine_time += times.tolist()
        coarse+= these_coarse[sample_mask][add_filter].tolist()

    channel= np.array(channel).flatten()
    card = np.array(card).flatten()
    charge = np.array(charge).flatten()
    fine_time = np.array(fine_time).flatten()
    coarse = np.array(coarse).flatten()
    time =  ((coarse + fine_time)*COUNTERS) #  % PERIOD

    slot_id, pmt_pos = get_info(card, channel)

    add_mask = np.array([sid not in [57, 58, 81] for sid in slot_id]).flatten()
    print("Cutting {} hits".format(np.sum(1-add_mask.astype(int))))


    time = time - np.min(time)
    all_data = np.array([
        time[add_mask], charge[add_mask], slot_id[add_mask], pmt_pos[add_mask]
    ]).T 

    all_data = sorted(all_data, key=lambda x:x[0], reverse=False) # we specifically list these in reverse time order! 

    print("Saving file")
    np.savetxt("hits.csv", all_data, delimiter=", ")
    return all_data


if __name__=="__main__":
    print(sys.argv[1])
    all_data = reader(sys.argv[1])
