import pandas as pd 
import numpy as np
import sys 
import os 
import json 
from glob import glob 

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
    print(all_files)
    for fn in all_files:
        print("Reading File {}".format(fn))
        data = pd.read_parquet(fn)

        card += np.array(data["card_id"][:]).tolist()
        channel += np.array(data["chan"][:]).tolist()
        charge += np.array(data["charge"][:]).tolist()
        fine_time += np.array(data["fine_time"][:]).tolist()
        coarse += np.array(data["coarse"][:]).tolist()

        print(len(coarse), "hits")

    card = np.array(card)
    channel= np.array(channel)


    keys = (100*card)+channel 

    mask = np.array([str(key) in pmt_data for key in keys])

    card = card[mask]
    channel = channel[mask]
    charge = np.array(charge)[mask]
    fine_time = np.array(fine_time)[mask]
    coarse = np.array(coarse)[mask]

    slot_id, pmt_pos = get_info(card, channel)

    time = (coarse*8 +  fine_time/8196.)
    time = time - np.min(time)
    mask = charge>30

    print(np.shape(time))
    
    all_data = np.array([
        time[mask], charge[mask], slot_id[mask], pmt_pos[mask]    
    ]).T 

    all_data = sorted(all_data, key=lambda x:x[0], reverse=False)

    print("Saving file")
    np.savetxt("hits.csv", all_data, delimiter=", ")



if __name__=="__main__":
    print(sys.argv[1])
    data = reader(sys.argv[1])