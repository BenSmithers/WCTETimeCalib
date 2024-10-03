

from WCTECalib.utils import C, N_WATER
from WCTECalib import get_pmt_positions

import numpy as np 
from scipy.optimize import minimize, basinhopping
import pandas as pd 
import os 

"""
    All of the fitting code here needs to allow for some timing offset.

    This...
        1. allows for the actual time of the event to float. We won't necessarily know at which coarse counter moment a ball or and LED flashes. We also won't know it's relative offset
        2. allows for only the relative timing offsets to be needed in doing this fit! 
"""

offset_dict = pd.read_csv(
    os.path.join(os.path.dirname(__file__),
    "..",
    "data",
    "calculated_offsets_lbmc.csv")
)
pmt_pos = np.transpose([
        offset_dict["X"], 
        offset_dict["Y"],
        offset_dict["Z"]
])  

def fit_hits(ids, sampled_times):

    trimmed= offset_dict[offset_dict["unique_id"].isin(ids)]
    times = sampled_times +np.array(trimmed["calc_offset"])
    
    positions = get_pmt_positions(ids)

    def metric(location_params):
        led_pos = location_params[:3]

        distances = np.sqrt(np.sum((positions - led_pos)**2, axis=1))
        predicted_times = (1e9)*distances*N_WATER/C  + location_params[3]
        return np.sum((times - predicted_times)**2)
    
    x0 = np.array([0.5,1,0.5, 0])
    bounds =[
        [-5,5],
        [-5,5],
        [-5,5],
        [-1e7, 1e7]
    ]
    options= {
        "eps":1e-10,
        "gtol":1e-10,
        "ftol":1e-10
    }
    res = basinhopping(metric, x0, niter=10, minimizer_kwargs={"options":options, "bounds":bounds})
    return res.x 