
from WCTECalib.times import generate_offsets, sample_balltime, DIFFUSER_ERR, BALL_ERR, mm, ns
from recover_ball import ballfit
from calculate_shifts import refit 
from WCTECalib.utils import get_color

import pandas as pd
import os 
import numpy as np
import matplotlib.pyplot as plt 


ball_errs = [0, 2*mm, 10*mm]
noises = [0.5, 1, 3]

def sample(ball_err, noise):
    generate_offsets()
    refit(noise, ball_err)



    offsets = pd.read_csv(os.path.join(os.path.dirname(__file__), "..","data","offsets.csv"))
    true_off = np.array(offsets["offsets"])

    fits = pd.read_csv(os.path.join(os.path.dirname(__file__), "..","data","calculated_offsets.csv"))
    fit_off = np.array(fits["calc_offset"])

    return fit_off - true_off 

for inoi, noise in enumerate(noises):
    for ib, ball_err in enumerate(ball_errs):

        offset_err = []
        bins = np.linspace(-2,2, 60)
        for i in range(40):
            offset_err+=sample(ball_err, noise).tolist()

        hist_data = np.histogram(offset_err, bins=bins)[0]
        hist_data=hist_data/np.sum(hist_data)

        ls = ["-", "--", '-.'][inoi]
        color = get_color(ib+1,4 ,"viridis")
        plt.stairs(hist_data, bins, color=color, ls=ls)
plt.xlabel("Offset Err [ns]",size=14)
plt.ylabel("Arb", size=14)

plt.plot([],[],ls="-", color='k',label="TTS {}ns".format(noises[0]))
plt.plot([],[],ls="--", color='k',label="TTS {}ns".format(noises[1]))
plt.plot([],[],ls="-.", color='k',label="TTS {}ns".format(noises[2]))
plt.plot([],[],color=get_color(0+1,4,"viridis" ), label="Ball Err 0.0mm")
plt.plot([],[],color=get_color(1+1,4 ,"viridis"), label="Ball Err 2mm")
plt.plot([],[],color=get_color(2+1,4 ,"viridis"), label="Ball Err 10mm")
plt.legend()
plt.savefig("./plots/offset_error.png",dpi=400)
plt.show()
