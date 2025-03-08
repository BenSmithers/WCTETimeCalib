

import numpy as np 
import os 
import pandas as pd 
from math import inf 
import matplotlib.pyplot as plt 
import json

TIME_MAGIC = 0.5703
AMP_MAGIC = 2.118 
DEBUG = True

cfd_raw_t = np.array([0.16323452713658082,
                0.20385733509493395,
                0.24339187740767365,
                0.2822514122310461,
                0.3208335490313887,
                0.35953379168152044,
                0.3987592183841288,
                0.4389432980060811,
                0.4805630068163285,
                0.5241597383052767,
                0.5703660640730557,
                0.6199413381955754,
                0.6738206794685682,
                0.7331844507933303,
                0.7995598000823612,
                0.874973724581176,
                0.9621917102137131,
                1.0301530251726216,
                1.0769047405430523,
                1.1210801763323819,
                1.1632345271365807])

# correction for the amplitude derived by summing the largest three adcs
amp_raw_t = np.array([2.0413475167493225, 2.0642014124776784, 2.0847238089021274, 2.1028869067818117, 2.118667914530039,
                        2.1320484585033723, 2.1430140317025583, 2.151553497195665, 2.1576586607668613, 2.1613239251470255,
                        2.162546035746829, 2.1613239251470255, 2.1576586607668617, 2.1515534971956654, 2.143014031702558,
                        2.1320484585033723, 2.118667914530039, 2.1028869067818117, 2.0847238089021274, 2.0642014124776784,
                        2.0413475167493225])

cfd_true_t =np.linspace(-0.5, 0.5, 21)

pathname = os.path.join(
    os.path.dirname(__file__),
    "..", "..",  
    "data",
    "laserball",
    "laser_ball_20241203111520_0_waveforms.parquet"
)

with open(os.path.join(os.path.dirname(__file__), "..","..","WCTECalib","geodata",'PMT_Mapping.json'), 'r') as file:
    pmt_data = json.load(file)["mapping"]

def get_info(card_id, channel):
    """
        returns slot_id, pmt_pos
    """

    # parse them! 

    keys = (100*card_id)+channel 
    long_form_id = np.array([pmt_data[str(key)] for key in keys])

    return long_form_id//100, long_form_id%100 
def heights(waveforms):
    diffs = np.max(waveforms,axis=1) - np.min(waveforms,axis=1)
    diffs = np.abs(diffs)

    bins = np.linspace(0, 300, 300)
    plt.hist(diffs, bins)
    plt.show()

def do_cfd2(waveforms):
    assert len(waveforms[0])==32, "Double-check the format of the waveforms file"
    fraction = 0.33
    nsample = len(waveforms[0])
    print(np.shape(waveforms))

    max_values = np.max(-1*waveforms, axis=1) 
    threshold_values = max_values * fraction
    cfd_points = np.zeros(waveforms.shape[0])   

    threshold_values = np.tile(threshold_values, nsample).reshape(nsample, len(threshold_values)).T

    crossings = np.diff(np.sign(-waveforms), axis=1)
    crossings[crossings<0]=0
    print(np.shape(crossings))
    crossings = np.where(crossings)
    print(np.shape(crossings))

    import sys 
    sys.exit()

def do_cfd(waveforms):
    #print("Running CFD")
    assert len(waveforms[0])==32, "Double-check the format of the waveforms file"
    
    nbins = 4
    nsample = len(waveforms[0])
    delay = 2
    multiplier = -2
    apply_correction = False

    baseline = np.mean(waveforms[:,:nbins], axis=1)

    # baseline is number of waves 
    baseline = np.tile(baseline,nsample).reshape(nsample, len(baseline)).T

    # SLOW
    sort_vals = np.sort(waveforms, axis=1)


    amplitudes = sort_vals[:, -1] + sort_vals[:, -2] + sort_vals[:, -3]
    amplitudes = amplitudes.astype(float)

    # converted to positive-going pulses
    trimmed =(waveforms[:, delay:] -baseline[:, delay:] ) + multiplier*(waveforms[:, :-delay] - baseline[:, delay:])
    trimmed = trimmed.astype(float)

    # now, find the index with the largest step that goes from + to -
    
    maxes = np.argmax(waveforms, axis=1)
    #plt.hist(maxes, np.arange(3.5, 11.51, 1))
    #plt.show()
    
    offsets = 5


    
    crossings = []
    # SLOW 
    for i in range(len(trimmed)):

        #indx = np.argwhere(np.diff(np.sign(trimmed[i][offsets:])))
        
        crossings.append(np.argmax(np.diff(trimmed[i]))+1 )
        continue
        if len(indx)==0:
            crossings.append(1)
            #crossings.append(np.argwhere(np.diff(np.sign(trimmed[i])))[0] + 1 )
        else:
            crossings.append(int(indx[0] + offsets + 1))
    
    crossings = np.array(crossings).flatten()
    

    times = np.zeros(len(crossings))

    # okay now apply the CFD 
    xmin = crossings-1
    xmax = crossings

    ymin = trimmed[np.arange(len(trimmed)), crossings-1]
    ymax =  trimmed[np.arange(len(trimmed)), crossings]

    slope = (xmax-xmin) / (ymax-ymin) 

    good_mask = slope>0
    # y0 = slope*x0 + int
    x_interp = xmin -  slope * ymin 
    
    delta = x_interp - (offsets +1)

    in_bounds = np.logical_and(cfd_raw_t[0] < delta, delta < cfd_raw_t[-1])

    shift_up = delta +1 
    shift_down = delta-1 

    shift_up_good = np.logical_and( np.logical_not(in_bounds), np.logical_and(cfd_raw_t[0] < shift_up, shift_up < cfd_raw_t[-1]))
    shift_down_good = np.logical_and( np.logical_not(in_bounds), np.logical_and(cfd_raw_t[0] < shift_down, shift_down < cfd_raw_t[-1]))
    all_bad = np.logical_not(np.logical_or(in_bounds, np.logical_or(shift_up_good, shift_down_good)))

    
    #apply_correction
    if apply_correction:


        times[in_bounds] = offsets + np.interp(delta[in_bounds], cfd_raw_t, cfd_true_t)
        times[shift_up_good] = offsets + 1 + np.interp(shift_up[shift_up_good], cfd_raw_t, cfd_true_t)
        times[shift_down_good] = offsets -1 + np.interp(shift_down[shift_down_good], cfd_raw_t, cfd_true_t)
    else:
        times = offsets  + delta

    times[all_bad] = x_interp[all_bad] - TIME_MAGIC
    amplitudes[all_bad] = amplitudes[all_bad]/AMP_MAGIC

    amplitudes[np.logical_not(all_bad)] /= np.interp(times[np.logical_not(all_bad)], cfd_true_t, amp_raw_t )

    times -= offsets

    #good_mask = np.logical_and(good_mask, maxes>=11)
    #good_mask = np.logical_and(good_mask, np.logical_not(in_bounds))
    #good_mask = np.logical_and(good_mask, np.logical_not(bad_sign))

    if DEBUG:
        i = 0
        plotted = 0
        while True:
            if  maxes[i]>=11 and times[i]>0.6:
                good_mask[i] = False
                #print(x_interp[i], delta[i])
                plt.clf()
                
                #trim_cros =np.argwhere(np.diff(np.sign(trimmed[i])))
                #plt.plot(crossings[i]+delay, trimmed[i][crossings[i]], 'rd', label="Crossings")

                plt.plot(range(len(waveforms[i])), waveforms[i], label="Waveform")
                plt.plot(np.array(range(len(trimmed[i]))) + delay, trimmed[i], label="Processed")

                plt.plot([xmin[i]+delay, xmax[i]+delay], [ymin[i], ymax[i]], label="Crossing", color="green")
                plt.grid(which='major', alpha=0.5)
                plt.xlabel("Coarse Counter", size=14)
                plt.ylabel("ADC", size=14)
                plt.legend()
                plt.tight_layout()
                plt.show()
                plotted+=1
            i+=1 
            
            if plotted>10:
                break

    
    return times[good_mask], amplitudes[good_mask], baseline[good_mask], good_mask, maxes[good_mask]


if __name__=="__main__":

    print("Reading file")
    import sys 

    # channel 6, card 14 
    from glob import glob 

    sus = [46]
    all_files = glob(sys.argv[1] + "/*waveforms.parquet")
    for filename in all_files:
        
        plt.clf()
        data = pd.read_parquet(filename)
        cards = np.array(data["card_id"])
        chan = np.array(data["chan"])
        print("Card : {}".format(cards[0]))
        
        #keys = (100*cards)+chan 
        #mask = np.array([str(key) in pmt_data for key in keys])


        mask = chan==6
        card = cards[mask]
        if card[0]!=75:
            continue
        channel = chan[mask]


        slot_id, pmt_pos = get_info(card, channel)

        do = np.logical_and(slot_id==0, pmt_pos==0)
        data = data["samples"][mask]

        print("Extracting Waveforms")
        waveforms = []
        for iw, wave in enumerate(data):
            if len(wave)==32 and np.max(wave)>100: 
                waveforms.append(wave)
                #plt.stairs(wave,range(33), alpha=0.1, color='k')
        waveforms= -1*np.array(waveforms)
        #heights(waveforms)
        if len(waveforms)==0:
            continue
        times, amplitudes,baseline, filter, maxes = do_cfd2(waveforms)



        n_bad = np.sum(np.logical_or(times>1, times<-1).astype(int))

        
        print(n_bad)
        ten = maxes<=10
        histlow = np.histogram(times[ten], np.linspace(-1, 1, 200))[0]
        histhi = np.histogram(times[np.logical_not(ten)], np.linspace(-1, 1, 200))[0]

        plt.stairs(histlow, np.linspace(-1, 1, 200), color='green', fill=True, label="<=10", alpha=0.75)
        plt.stairs(histhi+histlow, np.linspace(-1, 1, 200), color='red', fill=True, label="11+", alpha=0.75, baseline=histlow)
        plt.legend()
        plt.xlabel("CFD Time [CC]", size=14)
        plt.ylabel("N Waveforms", size=14)
        plt.show()

