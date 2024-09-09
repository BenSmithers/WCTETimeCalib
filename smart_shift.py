

from calculate_shifts import refit
from WCTECalib.fitting import fit_hits
from WCTECalib.utils import mm, N_WATER, C
from WCTECalib.geometry import get_pmt_positions
from WCTECalib.times import sample_balltime

import numpy as np
from scipy.optimize import minimize, basinhopping

def specific_fitter(ids, offset_times):
    positions = get_pmt_positions(ids)
    def metric(location_params):
        led_pos = location_params[:3]

        distances = np.sqrt(np.sum((positions - led_pos)**2, axis=1))
        predicted_times = (1e9)*distances*N_WATER/C  + location_params[3]
        return np.sum((offset_times - predicted_times)**2)
    x0 = np.array([0.05,0.5,0.1, 0])
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
    from copy import deepcopy

    t1_flash_loc = np.array([0,0.5,0])
    t2_flash_loc = np.array([0,0.5,0])

    f1, throw_one = refit(noise=0.0, central_ball_loc=t1_flash_loc,ball_err=100*mm, mu=10,n_flash=40)
    f2, throw_two = refit(noise=0.0, central_ball_loc=t2_flash_loc, ball_err=100*mm,mu=10,n_flash=40)

    new_location = np.array([0.1, -0.2, 0.3])

    obs_ids, sample_times, mus = sample_balltime(0, new_location,0.0, mu=10)

    pmt_pos = get_pmt_positions(obs_ids)

    # FIRST, we fit this according to each fit throw 

    set_1_times = deepcopy(sample_times) + deepcopy(throw_one[obs_ids])
    set_2_times = deepcopy(sample_times) + deepcopy(throw_two[obs_ids])

    throw_one_fit = specific_fitter(obs_ids, set_1_times)
    throw_two_fit = specific_fitter(obs_ids, set_2_times)

    print("fit1 err {}".format(np.sqrt(np.sum((throw_one_fit[:3] - new_location )**2))*1000))
    print("fit2 err {}".format(np.sqrt(np.sum((throw_two_fit[:3] - new_location )**2))*1000))
    

    distance_1 = pmt_pos - t1_flash_loc
    t1nofudge = np.sqrt(np.sum(distance_1**2, axis=1))*(1e9)*N_WATER/C

    distance_2 = pmt_pos - t2_flash_loc
    t2nofudge = np.sqrt(np.sum(distance_2**2, axis=1))*(1e9)*N_WATER/C

    def get_timing_fudges(location_params):
        throw_one_fudge = location_params[:3]
        throw_two_fudge = location_params[3:6]
        

        t1distance = pmt_pos - t1_flash_loc - throw_one_fudge
        t1fudge = np.sqrt(np.sum(t1distance**2, axis=1))
        t1_time_changes = (1e9)*t1fudge*N_WATER/C - t1nofudge


        t2distance = pmt_pos - t2_flash_loc - throw_two_fudge
        t2fudge = np.sqrt(np.sum(t2distance**2, axis=1))
        t2_time_changes = (1e9)*t2fudge*N_WATER/C - t2nofudge

        return t1_time_changes, t2_time_changes

    def metric(location_params):
        t1_time_changes, t2_time_changes = get_timing_fudges(location_params)

        throw_three_pos = location_params[6:9]
        flash_one = location_params[9]

        distances = np.sqrt(np.sum((pmt_pos - throw_three_pos)**2, axis=1))
        predicted_times = (1e9)*distances*N_WATER/C 

        #print("{} - {}".format(np.sum((set_1_times + t1_time_changes - predicted_times -flash_one)**2), np.sum((set_2_times + t2_time_changes - predicted_times-flash_one)**2)))

        return  np.sum((set_1_times + t1_time_changes - predicted_times -flash_one)**2) + np.sum((set_2_times + t2_time_changes - predicted_times-flash_one)**2) 

    x0 = np.array([
        0.001, -0.001, 0.001, 
        -0.001, 0.001, -0.001,
        0.05, 0.5, 0.05, 
        np.mean(sample_times),
    ])
    bounds = [[-5,5]]*6 + [[-5,5]]*3 + [[-1e7, 1e7]]
    options= {
        "eps":1e-8,
        "gtol":1e-10,
        "ftol":1e-10
    }
    res = basinhopping(metric, x0, minimizer_kwargs={'options':options, 'bounds':bounds}, niter=20)
    print(res)

    fit_loc = res.x[6:9]

    print("Error on better fit: {}".format(np.sqrt(np.sum((fit_loc - new_location)**2))*1000 ))
    return res.x

if __name__=="__main__":
    result= main()

    print(result[6:])
