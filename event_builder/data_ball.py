"""
    Screw all of that other crap, let's do this directly and explicitly 
"""
import numpy as np 
import pandas as pd
from cfd.do_cfd import do_cfd, pmt_data, get_info
import matplotlib.pyplot as plt 
from scipy.stats import mode 
from scipy.optimize import minimize
from WCTECalib import df, get_pmt_positions
from WCTECalib.utils import ball_pos, second, C, N_WATER
import os 
from copy import deepcopy 
from tqdm import tqdm

#filename = "../data/laserball/laser_ball_20241203111520_0_waveforms.parquet"
#filename = "../data/ukli/ukli_diffuser__20241205175133_0_waveforms.parquet"
# ilename = "../data/laserball/laser_ball_20241205181433_0_waveforms.parquet"
filename = "../data/laserball/laser_ball_20241205183135_6_waveforms.parquet"
DEBUG = True
PERIOD = 262144
dfile = pd.read_parquet(filename)

# okay, so what we're doing is going through and grabbing all of the hits on all of the PMTS
channels = np.array(range(19))
mPMT_card = np.unique(dfile["card_id"])
print(mPMT_card)

REFRENCE_TIME = -1
def do_fit(raw_times, raw_wave, x0, double_fit= False):
    def metric(params):
        mu = params[1]
        sigma =params[2]

        presum =(raw_wave - params[0]*np.exp(-0.5*((raw_times - mu)/sigma)**2))**2
        #vlate = this_wave- params[1] > 2
        #presum[vlate]= 0.0 

        return np.nansum(presum)
    def doublemet(params):
        mu = params[1]
        sigma = params[2]
        amp2 = params[3]

        presum = (raw_wave - params[0]*np.exp(-0.5*((raw_times - mu)/sigma)**2) - amp2*np.exp(-0.5*((raw_times -mu -8)/sigma)**2) )**2
        return np.nansum(presum)
    
    bounds = [
            (0, np.inf),
            (min(raw_times), max(raw_times)),
            (0.2, 20)
        ]
    if double_fit:
        bounds.append((0, np.inf))
    res = minimize( 
        doublemet if double_fit else metric, x0, bounds=bounds
    )
    return res.x, res.fun

new_df = deepcopy(df)
new_df["calc_offset"] = np.nan*np.ones_like(new_df["unique_id"])
new_df["nhits"] = np.nan*np.ones_like(new_df["unique_id"])
new_df["offset_sigma"] = np.nan*np.ones_like(new_df["unique_id"])
ref_time = -1 
for card_id in tqdm(mPMT_card):
    one_good = False
    plt.clf()
    fig,axes= plt.subplots(5,4, sharex=True, sharey=True)
    for channel in channels:
        # 5 rows, 4 column 
        i_column = channel % 4 
        irow = channel // 4

        # check if this is an important one 
        key = 100*card_id + channel
        exists = str(key) in pmt_data
        slot_id = -1
        pmt_pos = -1
        if not exists:
            continue
        else:
            slot_id, pmt_pos = get_info(np.array([card_id,]), np.array([channel,]))
            slot_id = slot_id[0]
            pmt_pos = pmt_pos[0]
        # get the data 
        data_mask = np.logical_and(dfile["card_id"]==card_id, dfile["chan"]==channel)

        these_waves = dfile["samples"][data_mask]
        these_coarse = dfile["coarse"][data_mask]
        # filter out the ones that aren't the right number of samples 
        selected = []
        keep_mask = [] 
        for i, wave in enumerate(these_waves):
            if len(wave) == 32: # other lengths have problems 
                selected.append(wave)
                keep_mask.append(True)
            else:
                keep_mask.append(False)
        selected = -1*np.array(selected)

        if len(selected)<20:
            continue
        fine_time, amp, base, cfd_filter = do_cfd(selected)    
         
        if REFRENCE_TIME==-1:
            # find reference time 
            
            _coarse_cut =np.array( these_coarse[keep_mask][cfd_filter])
            _coarse_time = _coarse_cut*8 % PERIOD
            REFRENCE_TIME = mode(_coarse_time).mode
            print("Setting Reference time to {}".format(REFRENCE_TIME))
            #fig2, ax2= plt.subplots()
            #ax2.hist(_coarse_time, bins=np.linspace(REFRENCE_TIME-10000, REFRENCE_TIME+10000, 2000))
            #plt.show()
            
            

        coarse_cut =np.array( these_coarse[keep_mask][cfd_filter]) - REFRENCE_TIME
                
        coarse_time = coarse_cut*8 % PERIOD

        mode_time = mode(coarse_time).mode
        shift_time = (( coarse_cut+ fine_time)*8) % PERIOD

        # okay we have our time distributions now 
        
        # let's just look at the mean time then 
        mean_time = np.nanmean(shift_time)

        bins = np.linspace(mode_time - 20, mode_time+20, 60)
        bcenter = 0.5*(bins[1:] + bins[:-1])
        binned = np.histogram(shift_time, bins)[0]
        result, metric = do_fit(bcenter, binned, [np.max(binned),mode_time, 1.5])
        new_one = [result[0], result[1], result[2], 0.25]
        double1, metric1 = do_fit(bcenter, binned, new_one, True)
        new_one = [result[0], result[1]-8, result[2], 0.25]
        double2,metric2 = do_fit(bcenter, binned, new_one, True)


        if metric1<metric2:
            double = double1
        else:
            double = double2

        fine_x = np.linspace( mode_time - 20, mode_time +20, 2000)
        is_double = double[-1]/double[0]>0.10
        if is_double:
            fine_y = double[0]*np.exp(-0.5*((fine_x - double[1])/double[2])**2) + double[3]*np.exp(-0.5*((fine_x - double[1]-8)/double[2])**2)
        else :
            fine_y = result[0]*np.exp(-0.5*((fine_x - result[1])/result[2])**2)


        if DEBUG and True:
            axes[irow][i_column].plot(fine_x, fine_y, label="Fit")
            axes[irow][i_column].stairs(binned ,bins, label="Data")
            axes[irow][i_column].set_ylim([0, 500])
            #axes[irow][i_column].vlines([result[1], result[1]+8 ],0, max(fine_y), color='red')
            #axes[irow][i_column].legend()
        else:
            plt.figure()
            plt.plot(fine_x, fine_y, label="Fit")
            plt.stairs(binned ,bins, label="Data")
            plt.legend()
            plt.xlabel("Time [ns]", size=14)
            plt.ylabel("Counts", size=14)
            plt.show()

        if result[0]>10 and (not is_double):
            if int(slot_id)==75 and int(pmt_pos) ==0:
                ref_time = result[1]

            who = np.logical_and(new_df["mPMT_slot_id"] == slot_id, new_df["Chan"]==pmt_pos)
            new_df.loc[who, "calc_offset"] = result[1]
            new_df.loc[who, "offset_sigma"] =result[2]
            new_df.loc[who, "nhits"] = result[0]
            one_good = True
        else:
            continue
    if one_good:
        plt.savefig("./plots/card{}.png".format(card_id))


positions= get_pmt_positions(new_df["unique_id"])
distances = np.sqrt(np.sum( (positions - ball_pos)**2 , axis=1)) #predicted distances
pred_time = second*distances*N_WATER/C

new_df["calc_offset"] = new_df["calc_offset"] - ref_time - pred_time


new_df.to_csv(
    os.path.join(os.path.dirname(__file__),"..", "data","calculated_offsets_realdata.csv"),
    index=False
)