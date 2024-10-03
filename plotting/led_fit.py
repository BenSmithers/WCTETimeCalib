from WCTECalib.times import sample_leds
from WCTECalib import  get_led_positions, get_led_dirs, get_pmt_positions
from WCTECalib.fitting import fit_hits
from WCTECalib.utils import N_WATER, C

import numpy as np
import os 
import pandas as pd 
from scipy.optimize import minimize
from math import sqrt
# let's just choose a barrel LED 

#which = (9+12)*3  + 16*3
led_max = 317

offset_dict = pd.read_csv(
    os.path.join(os.path.dirname(__file__),
    "..",
    "data",
    "calculated_offsets.csv")
)


def sample():
    which = float(np.random.randint(0, 318))
    led_ar_pos = get_led_positions([which,])
    led_dir = get_led_dirs([which,]).T
    led_pos = led_ar_pos[0]

    ids, times, evts = sample_leds(which, mu=100)
    trimmed= offset_dict[offset_dict["unique_id"].isin(ids)]
    
    result = fit_hits(ids, times)

    fit_position = result[:3]

    pmt_pos = get_pmt_positions(ids)
    #print("vertex error : {}".format(pmt_pos - led_pos))
    distances = np.sqrt(np.sum((pmt_pos - fit_position)**2, axis=1))

    fit_times = (1e9)*distances*N_WATER/C + np.array(trimmed["calc_offset"]) + result[3]

    return ids, times - fit_times 


if True :
    import matplotlib.pyplot as plt 

    # calculate bands 
    if True:

        n_samples = 100000
        many_samp = {}
        for i in range(n_samples):
            ids, err = sample()
            for i in range(len(ids)):
                if ids[i] in many_samp:
                    many_samp[ids[i]].append(err[i])
                else:
                    many_samp[ids[i]] = [err[i],]

        # this data *might* be irregularly shaped, so we need to do this in an ugly way
        mean_err = []
        err_range = []
        range_2s = []
        range_3s = []
        for key in sorted(list(many_samp.keys())):
            range_2s.append(np.percentile(np.abs(many_samp[key]),68.2689492))
            range_3s.append(np.percentile(np.abs(many_samp[key]),99.7300204))
        range_2s = np.array(range_2s)
        range_3s = np.array(range_3s)
        mean_err = np.array(mean_err)
        ids = np.array(sorted(list(many_samp.keys())))



    else:
        range_2s = 1.50
        range_3s = 4.5 

        n_samples = 1
        many_samp = {}
        for i in range(n_samples):
            ids, err = sample()
            for i in range(len(ids)):
                if ids[i] in many_samp:
                    many_samp[ids[i]].append(err[i])
                else:
                    many_samp[ids[i]] = [err[i],]

        # this data *might* be irregularly shaped, so we need to do this in an ugly way
        mean = []   
        eom = []
        for key in many_samp.keys():
            mean.append(np.mean(many_samp[key]))
            eom.append(np.std(mean)/sqrt(n_samples))

    def get_color(n, colormax=3.0, cmap="viridis"):
        """
            Discretize a colormap. Great getting nice colors for a series of trends on a single plot! 
        """
        this_cmap = plt.get_cmap(cmap)
        return this_cmap(n/colormax)



    #twosig = np.percentile(np.abs(mean_err), 68.2689492)
    #threesig = np.percentile(np.abs(mean_err), 99.7300204)

    #colors = get_color(np.abs(mean), 6, "RdBu")
    plt.fill_between(ids, -range_3s, range_3s, label=r"3$\sigma$", color="green", alpha=0.2)
    plt.fill_between(ids, -range_2s, range_2s, label=r"2$\sigma$",color="yellow", alpha=0.2)
    if False:
        if n_samples==1:
            plt.errorbar(ids, mean, yerr= None, ecolor='k', capsize=5, ls='', marker='d', color='r')
        else:
            plt.errorbar(ids, mean, yerr= eom, ecolor='k', capsize=5, ls='', marker='d', color='r')
    plt.xlabel("PMT No", size=14)
    plt.ylabel("Difference [ns]", size=14)
    plt.title("{} Samples".format(n_samples),size=14)
    plt.legend()
    plt.savefig("./plots/regions_of_err_led_eom.png")
    plt.show()