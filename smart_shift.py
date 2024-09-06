

from calculate_shifts import refit
from WCTECalib.fitting import fit_hits
from WCTECalib.utils import mm, N_WATER, C
from WCTECalib.geometry import get_pmt_positions
from WCTECalib.times import sample_balltime

import numpy as np
from scipy.optimize import minimize

def specific_fitter(ids, offset_times):
    positions = get_pmt_positions(ids)
    def metric(location_params):
        led_pos = location_params[:3]

        distances = np.sqrt(np.sum((positions - led_pos)**2, axis=1))
        predicted_times = (1e9)*distances*N_WATER/C  + location_params[3]
        return np.sum((offset_times - predicted_times)**2)
    x0 = np.array([0.05,0.5,0.05, 0])
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
    res = minimize(metric, x0,options=options, bounds=bounds)
    return res.x 

def main():
    ids, throw_one = refit(noise=0.0, ball_err=100*mm)
    ids, throw_two = refit(noise=0.0, ball_err=100*mm)

    new_location = np.array([0.1, 0.2, 0.5])

    obs_ids, sample_times, mus = sample_balltime(0, new_location,0.0, 10)

    pmt_pos = get_pmt_positions(obs_ids)

    # FIRST, we fit this according to each fit throw 
    set_1_times = sample_times + throw_one[obs_ids]
    set_2_times = sample_times + throw_two[obs_ids]

    throw_one_fit = specific_fitter(obs_ids, set_1_times)
    throw_two_fit = specific_fitter(obs_ids, set_2_times)

    print("fit1 err {}".format(np.sqrt(np.sum((throw_one_fit[:3] - new_location )**2))*1000))
    print("fit2 err {}".format(np.sqrt(np.sum((throw_two_fit[:3] - new_location )**2))*1000))
    

    distance_1 = np.sqrt(np.sum((pmt_pos - throw_one_fit[:3])**2, axis=1))
    t1nofudge = distance_1*(1e9)*N_WATER/C

    distance_2 = np.sqrt(np.sum((pmt_pos - throw_two_fit[:3])**2, axis=1))
    t2nofudge = distance_2*(1e9)*N_WATER/C

    delta_one_x = distance_1.T[0]
    delta_one_y = distance_1.T[1]
    delta_one_z = distance_1.T[2]
    delta_two_x = distance_2.T[0]
    delta_two_y = distance_2.T[1]
    delta_two_z = distance_2.T[2]


    def metric(location_params):
        throw_one_fudge = location_params[:3]
        throw_two_fudge = location_params[3:6]
        throw_three_pos = location_params[6:9]
        flash_one = location_params[9]

        
        thisdx = delta_one_x + throw_one_fudge[0]
        thisdy = delta_one_y + throw_one_fudge[1]
        thisdz = delta_one_z + throw_one_fudge[2]
        t1fudge = np.sqrt(thisdx**2 + thisdy**2 + thisdz**2)
        t1_time_changes = (1e9)*(t1fudge - t1nofudge)*N_WATER/C

        thisdx = delta_two_x + throw_two_fudge[0]
        thisdy = delta_two_y + throw_two_fudge[1]
        thisdz = delta_two_z + throw_two_fudge[2]
        t2fudge = np.sqrt(thisdx**2 + thisdy**2 + thisdz**2)
        t2_time_changes = (1e9)*(t2fudge - t2nofudge)*N_WATER/C

        distances = np.sqrt(np.sum((pmt_pos - throw_three_pos)**2, axis=1))
        predicted_times = (1e9)*distances*N_WATER/C 

        return np.sum((set_1_times + t1_time_changes - predicted_times -flash_one)**2) + np.sum((set_2_times + t2_time_changes - predicted_times-flash_one)**2)

    x0 = np.array([
        0.01, 0.01, 0.01, 
        0.01, 0.01, 0.01,
        0.05, 0.5, 0.05, 
        5,
    ])
    print(len(x0))
    bounds = [[-5,5]]*9 + [[-1e7, 1e7]]
    print(len(bounds))
    options= {
        "eps":1e-10,
        "gtol":1e-10,
        "ftol":1e-10
    }
    res = minimize(metric, x0,options=options, bounds=bounds)
    return res.x 


if __name__=="__main__":
    result= main()
    print(result)