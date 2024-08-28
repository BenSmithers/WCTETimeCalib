"""
    Generate an MC time sample thing 
"""

from WCTECalib.utils import *
from WCTECalib.geometry import get_pmts_visible, get_led_positions, df
from WCTECalib.geometry import N_CHAN, N_MPMT, get_pmt_positions
from copy import deepcopy
import numpy as  np
import pandas as pd 
import os 
from math import exp 

def sample_leds(led_no, mu=1, noise=NOISE_SCALE):
    """
    Possible output from flashing LED number `led_no` 
    at a brightness where the nearest LED would see on average 1 pe 

    Returns a tuple with three arrays
        0 - an array of IDs specifying specific PMTs 
        1 - the times at which a pulse was seen in the PMT 
        2 - the number of PEs in that PMT 
    """
    true_data = pd.read_csv(os.path.join(
        os.path.dirname(__file__), "..","data","offsets.csv"
    ))


    

    led_ar_pos = get_led_positions([led_no,])
    led_pos = led_ar_pos[0]

    # we get the positions of the PMTs that are visible
    # and the distances to those PMTs 
    keep = get_pmts_visible(led_no) # unique IDs! 
    true_data = true_data[keep]
    positions = get_pmt_positions(true_data["unique_id"])

    distances = np.sqrt(np.sum((led_pos - positions)**2, axis=1))

    true_times = second*distances*N_WATER/C + np.array(true_data["offsets"])
    pert_times = np.random.randn(N_CHAN*N_MPMT)[keep]*noise + true_times

    # normalize this around ~3m 
    dmin= 3
    munot = mu/exp(-dmin/ABS_LEN)
    mueff = munot*np.exp(-distances/ABS_LEN)

    m_sample = np.random.poisson(mueff)

    keep_these = m_sample>0

    return np.array(true_data["unique_id"])[keep_these], pert_times[keep_these], m_sample[keep_these]

def sample_balltime(noise = NOISE_SCALE, ball=None, ball_pos_noise = True, ball_err = BALL_ERR, diff_err=DIFFUSER_ERR):
    """
        Uses the sampled "true" shifts from `mc_timedist` 
        to generate a pseudoexperimental result 
        for a laser ball flasher result for a ball at `ball_pos`

        ball should be in meters 

        We also add in some coarse counter offset to represent this being at a random point during a data-taking run. 
    """
    
    coarse_counter_offset = np.random.rand()*(1e6)

    faux_data = pd.read_csv(os.path.join(
        os.path.dirname(__file__), "..","data","offsets.csv"
    ))

    true_data = deepcopy(faux_data)

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

    return pert_times + coarse_counter_offset
 

def generate_offsets(offset=OFFSET_SCALE):
    mpmt_offset_sample = np.random.rand(N_MPMT)*offset
    mpmt_offset_sample = np.repeat(mpmt_offset_sample, N_CHAN)

    pert_times = mpmt_offset_sample + np.random.randn(N_CHAN*N_MPMT)*PMT_OFF

    from copy import deepcopy

    new_df = deepcopy(df)

    new_df["offsets"] = pert_times

    new_df.to_csv(
        os.path.join(os.path.dirname(__file__), "..", "data","offsets.csv"),
        index=False
    )


if __name__=="__main__":
    generate_offsets()
