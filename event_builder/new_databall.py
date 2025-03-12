from glob import glob 
import numpy as np 
import pandas as pd
from cfd.do_cfd import do_cfd, pmt_data, get_info
import matplotlib.pyplot as plt 
from scipy.stats import mode 
from scipy.optimize import minimize, basinhopping
from WCTECalib import df, get_pmt_positions
from WCTECalib.utils import ball_pos, second, C, N_WATER
import os 
from copy import deepcopy 
from tqdm import tqdm
from scipy.signal import find_peaks

PERIOD = 262144

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

channels = range(19)
def process(filename, reference_time = -1, docfd=False):
    dfile = pd.read_parquet(filename)
    time_bins = np.linspace(dfile["coarse"].min(), dfile["coarse"].max(), 500)


    one_good = False
    card_id = dfile["card_id"][0]
    print("Working on {}".format(card_id))

    offset_map = {}
    sigma_map = {}
    hits_map = {}

    plt.clf()
    fig,axes= plt.subplots(5,4, sharex=True, sharey=False)
    axes[4][3].tick_params(axis='both', which='both', labelsize=2)
    for channel in channels:
        i_column = channel % 4
        irow = channel // 4
        axes[irow][i_column].tick_params(axis='both', which='both', labelsize=2)

        key = 100*card_id + channel
        exists = str(key) in pmt_data
        slot_id = -1
        pmt_pos = -1
        if not exists:
            print("Unknown mPMT/PMT Combo")
            continue
        else:
            slot_id, pmt_pos = get_info(np.array([card_id,]), np.array([channel,]))
            slot_id = slot_id[0]
            pmt_pos = pmt_pos[0]
            

        unique_id = slot_id*19 + pmt_pos

        these_coarse = dfile["coarse"]

        TMIN = 0.05*these_coarse.max()
        TMAX = 0.90*these_coarse.max()
        data_mask = np.logical_and(dfile["chan"]==channel, np.logical_and(these_coarse>TMIN, these_coarse<TMAX))
        these_coarse = dfile["coarse"][data_mask]
        if docfd:
            these_waves = dfile["samples"][data_mask]

            selected = []
            keep_mask = [] 
            for i, wave in enumerate(these_waves):
                if len(wave) == 32 and np.max(wave)>100: # other lengths have problems 
                    selected.append(wave)
                    keep_mask.append(True)
                else:
                    keep_mask.append(False)
            selected = -1*np.array(selected)

            if len(selected)<20:
                offset_map[unique_id] = np.nan
                sigma_map[unique_id] = np.nan
                hits_map[unique_id] = np.nan 
                print("Not enough waveforms")
                continue

            fine_time, amp, base, cfd_filter = do_cfd(selected)
            these_coarse =  these_coarse[keep_mask][cfd_filter]
        else:
            fine_time = dfile["fine_time"][data_mask]/65536.0

            if len(fine_time)<20:
                offset_map[unique_id] = np.nan
                sigma_map[unique_id] = np.nan
                hits_map[unique_id] = np.nan 
                print("Not enough waveforms")
                continue
            

        if reference_time==-1:
            _coarse_time = these_coarse*8 % PERIOD
            reference_time = mode(_coarse_time).mode


        coarse_cut =np.array(these_coarse) - reference_time
        
        binned = np.histogram(fine_time + these_coarse, time_bins)[0]
        fract_lt = np.sum((binned>(0.25*np.max(binned))).astype(float))
        fract_lt /= (len(time_bins)-1)

        coarse_time = coarse_cut*8 % PERIOD

        mode_time = mode(coarse_time).mode
        if np.isnan(mode_time):
            offset_map[unique_id] = np.nan
            sigma_map[unique_id] = np.nan
            hits_map[unique_id] = np.nan 
            print("Bad mode")
            continue
        shift_time = (( coarse_cut+ fine_time)*8) % PERIOD

        # let's just look at the mean time then 

        bins = np.linspace(mode_time - 20, mode_time+20, 120)
        bcenter = 0.5*(bins[1:] + bins[:-1])
        binned = np.histogram(shift_time, bins)[0]


        

        result, metric = do_fit(bcenter, binned, [np.max(binned),bcenter[np.argmax(binned)], 1.5])


        fine_x = np.linspace( mode_time - 20, mode_time +20, 2000)
        

        fine_y = result[0]*np.exp(-0.5*((fine_x - result[1])/result[2])**2)
        
        #peaks = find_peaks(binned, height=0.95*result[0])

        cross = np.diff(np.sign(binned - 0.5*np.max(result[0])))
        cross[ cross<0] = 0
        cross = np.argwhere(cross)
        peaks = len(cross)
        

        #is_double = len(peaks[0])>1
        is_double = peaks>1
        if is_double or result[0]<50:
            color='red'
        else :
            color='green'
        
        if result[0]>50 and (not is_double):
            one_good= True
            offset_map[unique_id] = result[1]
            sigma_map[unique_id] = result[2] 
            hits_map[unique_id] = result[0]/fract_lt
        else:
            offset_map[unique_id] = np.nan
            sigma_map[unique_id] = np.nan
            hits_map[unique_id] = np.nan
            if is_double:
                print("double", peaks) #, peaks[1]["peak_heights"] )
            else:
                print("Bad fit")

        axes[irow][i_column].plot(fine_x, fine_y, label="Fit", color=color)
        axes[irow][i_column].stairs(binned ,bins, label="Data")
        axes[irow][i_column].set_ylim([0, 1.1*np.max(binned)])
        axes[irow][i_column].set_xlim([fine_x.min(), fine_x.max()])
        if is_double:
            axes[irow][i_column].vlines([ result[1]-16, result[1]-8, result[1]+8, result[1]+16], 0, result[0], color='red', alpha=0.5, ls='--')
        
    
    if one_good or int(card_id)==131:
        plt.savefig("./hit_plots/card{}.png".format(card_id), dpi=400)
        plt.close()
    return offset_map, sigma_map, hits_map, reference_time

if __name__=="__main__":
    import sys 
    folder = sys.argv[1]

    new_df = deepcopy(df)
    new_df["calc_offset"] = np.nan*np.ones_like(new_df["unique_id"])
    new_df["nhits"] = np.nan*np.ones_like(new_df["unique_id"])
    new_df["offset_sigma"] = np.nan*np.ones_like(new_df["unique_id"])


    print("Processing {}".format(folder))
    reference_time = -1 
    ref_time = -1
    use_cfd = False 
    if use_cfd:
        all_files = glob(folder + "/*waveforms.parquet")
    else:
        all_files = glob(folder + "/*hits.parquet")
    for file in all_files:

        offsets, sigmas, hits, ref = process(file, reference_time, docfd=use_cfd)
        if ref!=-1:
            reference_time = ref
        
        for uqid in offsets.keys():
            pos = uqid % 19
            slot = uqid // 19 
            if int(slot)==39 and int(pos) ==0: #formerly 49
                ref_time = offsets[uqid]
            who = np.logical_and(new_df["mPMT_slot_id"] == slot, new_df["Chan"]==pos)
            new_df.loc[who, "calc_offset"] = offsets[uqid]
            new_df.loc[who, "offset_sigma"] = sigmas[uqid]
            new_df.loc[who, "nhits"] = hits[uqid]

    positions= get_pmt_positions(new_df["unique_id"])
    distances = np.sqrt(np.sum( (positions - ball_pos)**2 , axis=1)) #predicted distances
    pred_time = second*distances*N_WATER/C

    new_df["calc_offset"] = new_df["calc_offset"] - pred_time -ref_time


    new_df.to_csv(
        os.path.join(os.path.dirname(__file__),"..", "data","calculated_offsets_realdata.csv"),
        index=False
    )
