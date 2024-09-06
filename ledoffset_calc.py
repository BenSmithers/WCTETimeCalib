import pandas as pd 
import os 
import numpy as np 
from WCTECalib.geometry import N_MPMT, N_CHAN, get_led_positions, get_pmt_positions
from WCTECalib.utils import N_WATER, C, convert_to_2d_offset, NOISE_SCALE
from WCTECalib.times import second, sample_leds
from copy import deepcopy
from tqdm import tqdm

import matplotlib.pyplot as plt 

DEBUG = True 

def refit(noise, mu=1):
    from WCTECalib.geometry import df

    n_flash = 20
    n_led = 317 

    counts = np.zeros(N_MPMT*N_CHAN,dtype=int)
    relative_offset_counter = np.reshape(counts,(len(counts), 1))
    relative_offset_counter = relative_offset_counter - relative_offset_counter.T 
    relative_offset_matrix = np.zeros(np.shape(relative_offset_counter))

    for led_id in tqdm(range(n_led)):
        led_pos = get_led_positions([led_id,])[0]

        for i_sample in range(n_flash):
            ids, times, mus = sample_leds(led_id, mu) 
            hit_positions = get_pmt_positions(ids)

            distances = np.sqrt(np.sum( (hit_positions - led_pos)**2 , axis=1)) #predicted distances
            pred_time = second*distances*N_WATER/C

            calculated_offset = times - pred_time 

            counts*=0 
            counts[ids]=1
            kt, kt = np.meshgrid(counts, counts)
            keep_mesh = kt*kt.T

            these_offsets= np.zeros(N_MPMT*N_CHAN)
            these_offsets[ids] = calculated_offset

            differences = convert_to_2d_offset(these_offsets, False)
            differences[np.logical_not(keep_mesh)] = 0.0
            relative_offset_matrix += differences
            relative_offset_counter= relative_offset_counter + keep_mesh
    
    mean_offsets = relative_offset_matrix/relative_offset_counter
    resulting_offsets = np.zeros(N_MPMT*N_CHAN)
    counts_1d = np.zeros(N_MPMT*N_CHAN)

    if DEBUG:
        plt.pcolormesh(range(len(mean_offsets)), range(len(mean_offsets)), mean_offsets)
        plt.show()

    print(np.sum(np.isnan(mean_offsets[0]).astype(int)))
    for i in range(N_MPMT*N_CHAN):
        mask = np.logical_not(np.isnan(mean_offsets[i]))
        if i==0:
            resulting_offsets[mask]+=mean_offsets[i][mask]
            counts_1d[mask]+=mask.astype(int)[mask]
        else:

            resulting_offsets[mask] += mean_offsets[i][mask] + resulting_offsets[i-1]/counts_1d[i-1]
            counts_1d += mask.astype(int)
            # okay, now, these are going to be the offsets _relative_ to 

    final = resulting_offsets/counts_1d

    if DEBUG:
        plt.pcolormesh(range(len(mean_offsets)), range(len(mean_offsets)), convert_to_2d_offset(final,False))
        plt.show()
    
    new_df = deepcopy(df)

    new_df["calc_offset"] = final - final[0]

    new_df.to_csv(
        os.path.join(os.path.dirname(__file__), "data","calculated_offsets.csv"),
        index=False
    )
    return np.array(df["unique_id"]), final

if __name__=="__main__":
    refit(1.0*NOISE_SCALE)