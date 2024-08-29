import pandas as pd 
import os 
import numpy as np 
from WCTECalib.utils import ball_pos, C, N_WATER, NOISE_SCALE, convert_to_2d_offset
from WCTECalib.times import sample_balltime, second, BALL_ERR
from copy import deepcopy
from tqdm import tqdm
"""
    Calculates offsets relative to mPMT 0 and PMT 0
"""

def refit(noise, ball_err=  BALL_ERR, mu=1):
    from WCTECalib.geometry import df, N_CHAN, get_pmt_positions, N_MPMT


    samples = {}
    
    ball_pos_err = np.random.randn(3)*ball_err
    
    counts = np.zeros(N_MPMT*N_CHAN,dtype=int)
    relative_offset_counter = np.reshape(counts,(len(counts), 1))
    relative_offset_counter = relative_offset_counter - relative_offset_counter.T 
    relative_offset_matrix = np.zeros(np.shape(relative_offset_counter))

    for i in tqdm(range(400)):


        # ball time is offset by a bit
        # time is biased by ball error 
        ids, t_meas, npe = sample_balltime(noise=noise, ball=ball_pos+ball_pos_err, ball_pos_noise=False, diff_err=False, mu=mu)
        positions = get_pmt_positions(ids)

        distances = np.sqrt(np.sum( (positions - ball_pos)**2 , axis=1)) #predicted distances
        pred_time = second*distances*N_WATER/C
        
        calculated_offset =t_meas - pred_time 


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
    


    new_df = deepcopy(df)

    new_df["calc_offset"] = mean_offsets[0]

    new_df.to_csv(
        os.path.join(os.path.dirname(__file__), "data","calculated_offsets.csv"),
        index=False
    )
    return ball_pos+ball_pos_err

if __name__=="__main__":
    refit(NOISE_SCALE)