import pandas as pd 
import numpy as np 
import sys 
import os 
import json 

import matplotlib.pyplot as plt 



NS = 1.0 
COUNTERS = 8*NS
if False:
    # Open and read the JSON file
    with open(os.path.join(os.path.dirname(__file__), "..","WCTECalib","geodata",'PMT_Mapping.json'), 'r') as file:
        pmt_data = json.load(file)["mapping"]

    def get_info(card_id, channel):
        """
            returns slot_id, pmt_pos
        """
        long_form_id = pmt_data[str((100*card_id)+channel)]
        return long_form_id//100, long_form_id%100 

def reader(filename):
    print("Reading File {}".format(filename))
    data = pd.read_parquet(filename)


    card = np.array(data["card_id"][:])
    channel = np.array(data["chan"][:])
    charge = np.array(data["charge"][:])
    fine_time = np.array(data["fine_time"][:])
    coarse = np.array(data["coarse"][:])


    time = (coarse +  fine_time/8196.)*COUNTERS
    time = time - np.min(time)

    mask = charge>30

    tbin = np.linspace(0, max(time), 1000)
    tdata = np.histogram(time, tbin)[0]
    plt.stairs(tdata, tbin)
    plt.show()
    print("Spans {} ns".format(np.max(time) -np.min(time)))
    data_packet = np.array([
        time[mask], charge[mask], card[mask], channel[mask]
    ]).T

    print("Sorting events by time")
    data_packet = np.array(sorted(data_packet, key=lambda x:x[0]))
    return data_packet.T


def build_events(data_packet):
    print("Running Event Builder")

    WINDOW_WIDTH = 1000*NS
    HIT_THRESH=400*3/5 # unitless, number of hits

    hit_counter = 0
    hit_id = 0 
    all_times = data_packet[0] 
    event_id = np.ones_like(all_times)*-1

    level = 0
    percents = np.linspace(0, 1, 100)*len(all_times)
    print("counting over {} hits".format(len(all_times)))
    in_event = False
    while hit_id<len(all_times):
        if hit_id>percents[level]:
            level+=1 
            print("{:.2f}% done - event no {}".format(100*hit_id/len(all_times), hit_counter))

        now = all_times[hit_id]
        mask = np.logical_and( all_times>now-WINDOW_WIDTH, all_times<now+WINDOW_WIDTH)
        if np.sum(mask)>HIT_THRESH:
            in_event = True 
            print("Event Started")
            if hit_counter!=event_id[hit_id]:
                # this is a new hit, step up the hit counter 
                hit_counter+=1 

            event_id[mask]=hit_counter

            hit_id = np.argwhere(mask)[-1][0]
        else:
            if in_event:
                print("event ended")
            in_event = False

        hit_id += 1

    is_a_hit = event_id!=-1
    print("Found {} events".format(hit_counter))
    save_data = {
        "times":[], 
        "charge":[],
        "pmtid":[]
    }


    all_hitids = np.unique(event_id[is_a_hit])
    for hit_id in all_hitids:
        mask = event_id==hit_id 
        save_data["times"].append(all_times[mask].tolist())
        save_data["charge"].append(data_packet[1][mask].tolist())




    outfile = os.path.join(
        os.path.dirname(__file__),
        "..",
        "data",
        "laserball_data_raw.json"
    )
    _obj = open(outfile, 'wt')
    json.dump(save_data, _obj)
    _obj.close()

    


if __name__=="__main__":
    data = reader(sys.argv[1])
    build_events(data)
