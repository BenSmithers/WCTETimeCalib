import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
from glob import glob 
from WCTECalib.utils import get_color

from cfd.do_cfd import pmt_data, get_info
from scipy.optimize import minimize 

def fit_func(chargeval, counts):
    def metric(params):
        amp = params[0]
        mean = params[1]
        width = params[2]

        return np.sum((counts - amp*np.exp(-0.5*((chargeval - mean)/width)**2 ))**2)

    bounds =[
            (0, np.inf), 
            (0, 500),
            (1e-3, 500)
        ]
    x0 = [
        np.max(counts),
        chargeval[np.argmax(counts)],
        50
    ]
    res = minimize( metric, x0=x0, bounds=bounds)
    return res.x

if __name__=="__main__":
    import sys 
    folder = sys.argv[1]


    print("Processing {}".format(folder))
    reference_time = -1 
    ref_time = -1
    use_cfd = False 
    cbins = np.linspace(0, 1000, 100)
    ccenter = 0.5*(cbins[1:] + cbins[:-1])
    if use_cfd:
        all_files = glob(folder + "/*waveforms.parquet")
    else:
        all_files = glob(folder + "/*hits.parquet")
    peaks = []
    for file in all_files:
        print("Reading ",file)
        data = pd.read_parquet(file)
        charges = data["charge"]
        card_id = np.array(data["card_id"])
        channel = np.array(data["chan"])
        plt.clf()
        for chan in range(19):
            

            key = 100*card_id[0] + chan
            exists = str(key) in pmt_data
            if not exists:
                continue


            slot_id, pmt_pos = get_info(np.array([card_id[0]]), np.array([chan,]))
            slot_id = slot_id[0]
            pmt_pos = pmt_pos[0]
            unique_id = 19*slot_id + pmt_pos

            mask = pmt_pos==channel
            binned = np.histogram(charges[mask], cbins)[0]
            binned = binned.astype(float)

            
            
            fit = fit_func(ccenter, binned)
            fit[0]/=np.sum(binned)
            binned/=np.sum(binned)
            peaks.append( fit[1] )
            xfine = np.linspace(0, 1000, 2000)
            yfine = fit[0]*np.exp(-0.5*((xfine-fit[1])/fit[2])**2)
            plt.stairs(binned, cbins, label="Pos {}".format(chan),color=get_color(chan+3, 25))
            #plt.plot(xfine, yfine, color='gray', ls='--')
        plt.xlabel("Charge [ADC]", size=14)
        plt.ylabel("Noralized", size=14)
        plt.savefig("./plots/charge/mpmt_charges_card_{}.png".format(card_id[0]))
    
    plt.clf()
    nubins =np.linspace(0, 350, 50)
    histit = np.histogram(peaks, nubins)[0]
    plt.stairs(histit, nubins, fill=True)
    plt.xlabel("Q1 Gain", size=14)
    plt.ylabel("Count", size=14)
    plt.savefig("./plots/charge/gain_distribution.png", dpi=400)