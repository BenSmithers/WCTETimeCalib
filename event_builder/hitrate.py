import numpy as np 
import pandas as pd
from cfd.do_cfd import do_cfd, pmt_data, get_info
import matplotlib.pyplot as plt 
from glob import glob 
from WCTECalib.utils import get_color

from tqdm import tqdm

#filename = "../data/laserball/laser_ball_20241205183135_6_waveforms.parquet"

def bin_data(filename, time_bins=None):
    dfile = pd.read_parquet(filename) 
    if time_bins is None:
        #time_bins = np.linspace(dfile["coarse"].max()*0.05, dfile["coarse"].max()*0.90, 50)
        time_bins = np.linspace(dfile["coarse"].min(), dfile["coarse"].max()/5, 50)
    card_id = dfile["card_id"][0]

    rawtime =dfile["coarse"]
    these_waves = dfile["samples"]

    if False:
        selected = []
        keep_mask = [] 
        for i, wave in enumerate(these_waves):
            if len(wave) == 32: # other lengths have problems 
                selected.append(wave)
                keep_mask.append(True)
            else:
                keep_mask.append(False)

        selected = -1*np.array(selected)
        fine_time, amp, base, cfd_filter = do_cfd(selected)

    final_times =  rawtime# [keep_mask]

    binned = np.histogram(final_times, time_bins)[0]

    return card_id, binned, time_bins

def old_mode(filename, ):
    dfile = pd.read_parquet(filename)

    # okay, so what we're doing is going through and grabbing all of the hits on all of the PMTS
    channels = np.array(range(19))
    mPMT_card = np.unique(dfile["card_id"])
    print(mPMT_card)

    all_binned_data = []

    rawtime =dfile["coarse"]

    time_bins = np.linspace(dfile["coarse"].min(), dfile["coarse"].max(), 1000)
    tcenter = 0.5*(time_bins[:-1] + time_bins[1:])
    twide = time_bins[1:] - time_bins[:-1] 
    #print("Working..",end='')
    index = 0
    for card_id in tqdm(mPMT_card):
        print("Card {}".format(card_id))
        data_mask =  dfile["card_id"]==card_id
        these_waves = dfile["samples"][data_mask]
        these_coarse = dfile["coarse"][data_mask]

        channel = 0
        key = 100*card_id + channel
        exists = str(key) in pmt_data
        slot_id = -1
        pmt_pos = -1
        if not exists:
            continue
        else:
            slot_id, pmt_pos = get_info(np.array([card_id,]), np.array([channel,]))
            slot_id = slot_id[0]
            pmt_pos = pmt_pos[0]


        selected = []
        keep_mask = [] 
        for i, wave in enumerate(these_waves):
            if len(wave) == 32: # other lengths have problems 
                selected.append(wave)
                keep_mask.append(True)
            else:
                keep_mask.append(False)
        selected = -1*np.array(selected)

        if len(selected)<20:
            continue
        #fine_time, amp, base, cfd_filter = do_cfd(selected)
        #print('.', end='')
        binned = np.histogram( these_coarse, time_bins)[0]
        #binned = np.histogram(fine_time + these_coarse[keep_mask][cfd_filter], time_bins)[0]
        #binned = (1e-3)*binned / (twide*1e-9)

        start = np.mean(binned[:5])
        end = np.mean(binned[-5:])
        middle = np.mean(binned[np.argmax(binned)-2:np.argmax(binned)+3])

        #print(start, middle, end)

        check1 = np.abs((start - middle)/middle)>0.1
        check2 = np.abs((end - middle)/end)>0.1
        #print(check1 ,check2 )
        if True: #check1 and check2:
            print("Bad one!")
            plt.stairs(binned, 8*time_bins/(60e9), color=get_color(card_id, 120,"inferno"), alpha=1.0, fill=True)
            plt.xlabel("Time [min]")
            #plt.yscale('log')
            plt.ylabel("Hits [unitless]")
            plt.tight_layout()
            plt.show()
        index += 1


if __name__=="__main__":
    import sys  
    folder = sys.argv[1]

    print("Processing {}".format(folder))
    reference_time = -1 
    all_files = glob(folder + "/*_waveforms.parquet")

    time_bins = None 
    for file in tqdm(all_files):
        plt.clf()
        
        card_id, binned_data, _time_bins = bin_data(file, time_bins)
        if int(card_id)>=130:
            continue
        if time_bins is None:
            time_bins = _time_bins
        print("Card: {}".format(card_id))
        time_centers = 0.5*(time_bins[1:] + time_bins[:-1])
        plt.stairs(binned_data, 8*time_bins/(60e9), color=get_color(card_id, 120,"inferno"), alpha=1.0, fill=True)
        #plt.plot(8*time_centers/(60e9), binned_data, color=get_color(card_id, 120,"inferno"), alpha=0.1)
        plt.xlabel("Time [minutes]",size=14)
        plt.ylabel("Hits", size=14)
        plt.tight_layout()
        plt.savefig("./plots/rate_card{}.png".format(card_id), dpi=400)
        plt.show()
