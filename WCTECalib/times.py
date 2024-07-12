"""
    Generate an MC time sample thing 
"""

from WCTECalib.utils import *

def sample_balltime(noise = NOISE_SCALE, ball=None, ball_pos_noise = True, ball_err = BALL_ERR, diff_err=DIFFUSER_ERR):
    """
        Uses the sampled "true" shifts from `mc_timedist` 
        to generate a pseudoexperimental result 
        for a laser ball flasher result for a ball at `ball_pos`

        ball should be in meters 
    """
    from WCTECalib.geometry import N_CHAN, N_MPMT, get_pmt_positions

    import numpy as  np
    import pandas as pd 
    import os 

    true_data = pd.read_csv(os.path.join(
        os.path.dirname(__file__), "..","data","offsets.csv"
    ))

    positions = get_pmt_positions()

    if ball is None:
        ball = ball_pos #default position
        nominal = True
    else:
        nominal = False 


    ball_pos_shift = np.random.randn(3)*ball_err

    if not ball_pos_noise:
        ball_pos_shift*=0.0

    distances = np.sqrt(np.sum( (positions - ball+ball_pos_shift)**2 , axis=1))
    true_times = second*distances*N_WATER/C


    true_offsets = true_times + np.array(true_data["offsets"])

    sample_differr = np.random.randn()*diff_err

    pert_times = np.random.randn(N_CHAN*N_MPMT)*noise + true_offsets + sample_differr

    return pert_times
 

def generate_offsets(offset=OFFSET_SCALE):

    from WCTECalib.geometry import df, N_CHAN, N_MPMT

    import numpy as  np
    import os 



    mpmt_offset_sample = np.random.rand(N_MPMT)*offset
    mpmt_offset_sample = np.repeat(mpmt_offset_sample, N_CHAN)

    pert_times = mpmt_offset_sample + np.random.randn(N_CHAN*N_MPMT)*PMT_OFF

    from copy import deepcopy

    new_df = deepcopy(df)

    new_df["offsets"] = pert_times

    new_df.to_csv(
        os.path.join(os.path.dirname(__file__), "..", "data","offsets.csv")
    )


if __name__=="__main__":
    generate_offsets()
