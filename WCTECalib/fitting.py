

from WCTECalib.utils import C, N_WATER
from WCTECalib.geometry import get_pmt_positions

import numpy as np 
from scipy.optimize import minimize
import pandas as pd 
import os 

offset_dict = pd.read_csv(
    os.path.join(os.path.dirname(__file__),
    "..",
    "data",
    "calculated_offsets.csv")
)
pmt_pos = np.transpose([
        offset_dict["X"], 
        offset_dict["Y"],
        offset_dict["Z"]
])  


def fit_ball(pulse_times):
    def metric(ball_params):
        """
            Calculate a metric for the given ball position 
        """
        ball_pos = ball_params[:3]
        distances = np.sqrt(np.sum((pmt_pos - ball_pos)**2, axis=1))


        times = (1e9)*distances*N_WATER/C + np.array(offset_dict["calc_offset"]) + ball_params[3]

        return np.sum((times - pulse_times)**2)


    x0 = np.array([0.1, 0.5, 1.5, 0])
    bounds =[
        [-5,5],
        [-5,5],
        [0,5],
        [-50,50]
    ]
    options= {
        "eps":1e-10,
        "gtol":1e-10,
        "ftol":1e-10
    }
    res = minimize(metric, x0,options=options, bounds=bounds)
    return res.x 
 

def fit_led_result(ids, sampled_times):

    trimmed= offset_dict[offset_dict["unique_id"].isin(ids)]

    positions = get_pmt_positions(ids)

    def metric(location_params):
        led_pos = location_params[:3]

        distances = np.sqrt(np.sum((positions - led_pos)**2, axis=1))
        predicted_times = (1e9)*distances*N_WATER/C + np.array(trimmed["calc_offset"]) + location_params[3]
        return np.sum((sampled_times - predicted_times)**2)
    
    x0 = np.array([0,0,0, 0])
    bounds =[
        [-5,5],
        [-5,5],
        [-5,5],
        [-50,50]
    ]
    options= {
        "eps":1e-10,
        "gtol":1e-10,
        "ftol":1e-10
    }
    res = minimize(metric, x0,options=options, bounds=bounds)
    return res.x 