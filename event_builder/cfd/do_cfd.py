

import numpy as np 
import os 
import pandas as pd 
from math import inf 
import matplotlib.pyplot as plt 
from magic import *


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
    print("doming CFD")
    assert len(waveforms[0])==32, "Double-check the format of the waveforms file"
    nbins = 3
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

    # now, find the index with the largest step that goes from + to -
    diffs = trimmed[:,1:] - trimmed[:, :-1]
    not_swing = np.logical_or(diffs <= trimmed[:, :-1], trimmed[:, :-1]>0)
    #diffs[not_swing] = -1
    biggest_drop = np.argmax(diffs, axis=1)+1

    times = np.zeros(len(biggest_drop))
    times[np.isnan(biggest_drop)] = np.nan 
    amplitudes[np.isnan(biggest_drop)] =np.nan

    # okay now apply the CFD 
    xmin = biggest_drop-1
    xmax = biggest_drop

    ymin = trimmed[np.arange(len(trimmed)), biggest_drop-1]

    ymax =  trimmed[np.arange(len(trimmed)), biggest_drop]
    
    x_interp = xmin -  (xmax-xmin) / (ymax-ymin) * ymin 
    offset = 6
    delta = x_interp - offset 


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

        #times[all_bad] = x_interp[all_bad] - TIME_MAGIC
        amplitudes[all_bad] = amplitudes[all_bad]/AMP_MAGIC

        amplitudes[np.logical_not(all_bad)] = np.interp(times[np.logical_not(all_bad)], cfd_true_t, amp_raw_t )

    return times, amplitudes, baseline



if __name__=="__main__":
    print("Reading file")
    data = pd.read_parquet(pathname)["samples"]

    print("Extracting Waveforms")
    waveforms = []
    for wave in data:
        if len(wave)==32:
            waveforms.append(wave)
            #plt.bar(range(32),wave, alpha=0.1, color='k')
    waveforms= -1*np.array(waveforms)


    #heights(waveforms)
    times, amplitudes,baseline = do_cfd(waveforms)

    plt.hist(times, bins=np.linspace(0, 10, 100))
    plt.show()

