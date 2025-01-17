

import numpy as np 
import os 
import pandas as pd 
from math import inf 
import matplotlib.pyplot as plt 


TIME_MAGIC = 0.5703
AMP_MAGIC = 2.118 

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

cfd_true_t =np.array([-0.5 + 0.05*i for i in range(21)])

pathname = os.path.join(
    os.path.dirname(__file__),
    "..", "..",  
    "data",
    "laserball",
    "laser_ball_comp_20241203112510_5_waveforms.parquet"
)

def heights(waveforms):
    diffs = np.max(waveforms,axis=1) - np.min(waveforms,axis=1)
    diffs = np.abs(diffs)

    bins = np.linspace(0, 300, 300)
    plt.hist(diffs, bins)
    plt.show()

def do_cfd(waveforms):
    print("Running CFD")
    assert len(waveforms[0])==32, "Double-check the format of the waveforms file"
    
    nbins = 4
    nsample = len(waveforms[0])
    delay = 2
    multiplier = -2
    apply_correction = True

    baseline = np.mean(waveforms[:,:nbins], axis=1)

    # baseline is number of waves 
    baseline = np.tile(baseline,nsample).reshape(nsample, len(baseline)).T

    print("sorting samples")
    sort_vals = np.sort(waveforms, axis=1)

    


    amplitudes = sort_vals[:, -1] + sort_vals[:, -2] + sort_vals[:, -3]
    amplitudes = amplitudes.astype(float)

    # converted to positive-going pulses
    trimmed =(waveforms[:, delay:] -baseline[:, delay:] ) + multiplier*(waveforms[:, :-delay] - baseline[:, delay:])
    trimmed = trimmed.astype(float)

    

    # now, find the index with the largest step that goes from + to -
    diffs = trimmed[:,1:] - trimmed[:, :-1]
    not_swing = np.logical_or(diffs <= trimmed[:, :-1], trimmed[:, :-1]>0)
    #diffs[not_swing] = -1
    offset = 5
    #crossings = np.array([np.argwhere(np.diff(np.sign(trimmed[i][offset:])))[0] for i in range(len(trimmed))]) + offset +1
    crossings = []
    for i in range(len(trimmed)):
        indx = np.argwhere(np.diff(np.sign(trimmed[i][offset:])))
        if len(indx)==0:
            crossings.append(np.argwhere(np.diff(np.sign(trimmed[i])))[0] + 1 )
        else:
            crossings.append(indx[0] + offset + 1)
    
    crossings = np.array(crossings).flatten()
    
    biggest_drop = np.argmax(diffs, axis=1)+1

    times = np.zeros(len(biggest_drop))
    times[np.isnan(biggest_drop)] = np.nan 
    amplitudes[np.isnan(biggest_drop)] =np.nan

    # okay now apply the CFD 
    xmin = crossings-1
    xmax = crossings

    ymin = trimmed[np.arange(len(trimmed)), crossings-1]
    ymax =  trimmed[np.arange(len(trimmed)), crossings]

    
    x_interp = xmin -  (xmax-xmin) / (ymax-ymin) * ymin 
    
    delta = x_interp - (offset +1)
    


    if apply_correction:

        in_bounds = np.logical_and(cfd_raw_t[0] < delta, delta < cfd_raw_t[-1])

        shift_up = delta +1 
        shift_down = delta-1 

        shift_up_good = np.logical_and( np.logical_not(in_bounds), np.logical_and(cfd_raw_t[0] < shift_up, shift_up < cfd_raw_t[-1]))
        shift_down_good = np.logical_and( np.logical_not(in_bounds), np.logical_and(cfd_raw_t[0] < shift_down, shift_down < cfd_raw_t[-1]))
        all_bad = np.logical_not(np.logical_or(in_bounds, np.logical_or(shift_up_good, shift_down_good)))

        times[in_bounds] = offset + np.interp(delta[in_bounds], cfd_raw_t, cfd_true_t)
        times[shift_up_good] = offset - 1 + np.interp(shift_up[shift_up_good], cfd_raw_t, cfd_true_t)
        times[shift_down_good] = offset +1 + np.interp(shift_down[shift_down_good], cfd_raw_t, cfd_true_t)

        times[all_bad] = x_interp[all_bad] - TIME_MAGIC
        amplitudes[all_bad] = amplitudes[all_bad]/AMP_MAGIC

        amplitudes[np.logical_not(all_bad)] /= np.interp(times[np.logical_not(all_bad)], cfd_true_t, amp_raw_t )

    times -= offset

    if True:
        i = 0
        plotted = 0
        while True:
            if times[i]<0.3395 and times[i]>0.3373:
                #print(x_interp[i], delta[i])
                plt.clf()
                
                #trim_cros =np.argwhere(np.diff(np.sign(trimmed[i])))
                #plt.plot(crossings[i]+delay, trimmed[i][crossings[i]], 'rd', label="Crossings")

                plt.plot(range(len(waveforms[i])), waveforms[i], label="waveform")
                plt.plot(np.array(range(len(trimmed[i]))) + delay, trimmed[i], label="Convolved")

                plt.plot([xmin[i]+delay, xmax[i]+delay], [ymin[i], ymax[i]], label="Crossing", color="green")
                plt.grid(which='major', alpha=0.5)
                plt.legend()
                plt.show()
                plotted+=1 
            i+=1 
            
            if plotted>10:
                break


    return times, amplitudes, baseline


if __name__=="__main__":

    print("Reading file")
    data = pd.read_parquet(pathname)["samples"]

    print("Extracting Waveforms")
    waveforms = []
    for wave in data:
        if len(wave)==32:
            waveforms.append(wave)
            #plt.stairs(wave,range(33), alpha=0.1, color='k')
    waveforms= -1*np.array(waveforms)


    #heights(waveforms)
    times, amplitudes,baseline = do_cfd(waveforms)

    plt.hist(times, bins=np.linspace(-1, 1, 1000))
    plt.show()

