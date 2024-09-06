import pandas as pd 
import os 
import numpy as np 
from WCTECalib.utils import ball_pos, C, N_WATER, NOISE_SCALE, convert_to_2d_offset, mm
from WCTECalib.times import sample_balltime, second, BALL_ERR
from copy import deepcopy
from tqdm import tqdm
"""
    Calculates offsets relative to mPMT 0 and PMT 0
"""
DEBUG = False 
def refit(noise, ball_err=  BALL_ERR, mu=1):
    from WCTECalib.geometry import df, N_CHAN, get_pmt_positions, N_MPMT

    print("{} - {}".format(noise, ball_err))

    samples = {}
    
    ball_pos_err = np.random.rand(3)
    ball_pos_err /= np.sqrt(np.sum(ball_pos_err**2))
    ball_pos_err*=ball_err
    
    print("Ball offset: {} {} {}".format(ball_pos_err[0]*1000, ball_pos_err[1]*1000, ball_pos_err[2]*1000))

    offsets = np.linspace(-155, 155, 3600)
    binids = np.arange(-0.5, N_CHAN*N_MPMT+0.5, 1)
    all_bins = np.zeros(( len(binids)-1, len(offsets)-1))
    for i in tqdm(range(400)):


        # ball time is offset by a bit
        # time is biased by ball error 
        ids, t_meas, npe = sample_balltime(noise=noise, ball=ball_pos+ball_pos_err, diff_err=False, mu=mu)
        positions = get_pmt_positions(ids)

        if 0 not in ids:
            continue

        distances = np.sqrt(np.sum( (positions - ball_pos)**2 , axis=1)) #predicted distances
        pred_time = second*distances*N_WATER/C
        
        calculated_offset = t_meas - pred_time 
        calculated_offset = calculated_offset[0]-calculated_offset 

        all_bins += np.histogram2d(ids, calculated_offset, bins=(binids, offsets))[0]

    id_mesh, offset_mesh = np.meshgrid(0.5*(binids[1:] + binids[:-1]), 0.5*(offsets[1:] + offsets[:-1]))

    # calculate the mean of the distribution
    mean_offsets = np.sum(offset_mesh.T*all_bins, axis=1)/np.sum(all_bins, axis=1)
    
    if DEBUG:
        import matplotlib.pyplot as plt 
        plt.pcolormesh(binids, offsets, np.log(all_bins.T +1))
        plt.xlabel("ID")
        plt.ylabel("Offset time")
        plt.show()

    new_df = deepcopy(df)

    new_df["calc_offset"] = mean_offsets

    new_df.to_csv(
        os.path.join(os.path.dirname(__file__), "data","calculated_offsets.csv"),
        index=False
    )
    return np.array(df["unique_id"]), mean_offsets

if __name__=="__main__":
    refit(noise=0.0*NOISE_SCALE, ball_err=100*mm)