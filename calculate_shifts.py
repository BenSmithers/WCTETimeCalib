import pandas as pd 
import os 
import numpy as np 
from WCTECalib.utils import ball_pos, C, N_WATER, NOISE_SCALE
from WCTECalib.times import sample_balltime, second, BALL_ERR
from copy import deepcopy

def refit(noise, ball_err=  BALL_ERR):
    from WCTECalib.geometry_old import df, N_CHAN, get_pmt_positions, N_MPMT


    samples = []
    positions = get_pmt_positions()
    ball_pos_err = np.random.randn(3)*ball_err
    

    for i in range(400):


        # ball time is offset by a bit
        # time is biased by ball error 
        t_meas = sample_balltime(noise=noise, ball=ball_pos+ball_pos_err, ball_pos_noise=False, diff_err=False)
        

        distances = np.sqrt(np.sum( (positions - ball_pos)**2 , axis=1)) #predicted distances
        pred_time = second*distances*N_WATER/C

        #print(t_meas[:10], pred_time[:10])
        
        calculated_offset =t_meas - pred_time 
        samples.append(calculated_offset)

    new_df = deepcopy(df)

    new_df["calc_offset"] = np.mean(samples , axis=0)

    new_df.to_csv(
        os.path.join(os.path.dirname(__file__), "data","calculated_offsets.csv"),
        index=False
    )
    return ball_pos+ball_pos_err

if __name__=="__main__":
    refit(NOISE_SCALE)