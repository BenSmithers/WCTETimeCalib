
from WCTECalib.times import generate_offsets, sample_balltime, DIFFUSER_ERR, BALL_ERR, mm, ns
from calculate_shifts import refit 
from WCTECalib.utils import get_color

import pandas as pd
import os 
import numpy as np
import matplotlib.pyplot as plt 


ball_errs = [2*mm]
intensity = [0.3, 1, 10]

def sample(ball_err, noise, mu):
    generate_offsets()
    refit(noise, ball_err, mu)



    offsets = pd.read_csv(os.path.join(os.path.dirname(__file__), "..","data","offsets.csv"))
    true_off = np.array(offsets["offsets"])
    true_off-=true_off[0]
    true_off*=-1

    fits = pd.read_csv(os.path.join(os.path.dirname(__file__), "..","data","calculated_offsets.csv"))
    fit_off = np.array(fits["calc_offset"])

    return fit_off - true_off 

for inoi, mu in enumerate(intensity):
    for ib, ball_err in enumerate(ball_errs):

        offset_err = []
        bins = np.linspace(-2,2, 60)
        for i in range(20):
            offset_err+=sample(ball_err, 0.5, mu).tolist()

        hist_data = np.histogram(offset_err, bins=bins)[0]
        hist_data=hist_data/np.sum(hist_data)

        ls = ["-", "--", '-.'][ib]
        color = get_color(inoi+1,4 ,"viridis")
        plt.stairs(hist_data, bins, color=color, ls=ls)
plt.xlabel("Offset Err [ns]",size=14)
plt.ylabel("Arb", size=14)

#plt.plot([],[],ls="-", color='k',label="mu {}pe".format(intensity[0]))
#plt.plot([],[],ls="--", color='k',label="mu {}pe".format(intensity[1]))
#plt.plot([],[],ls="-.", color='k',label="mu {}pe".format(intensity[2]))
plt.plot([],[],color=get_color(0+1,4,"viridis" ), label="mu={}pe".format(intensity[0]))
plt.plot([],[],color=get_color(1+1,4 ,"viridis"), label="mu={}pe".format(intensity[1]))
plt.plot([],[],color=get_color(2+1,4 ,"viridis"), label="mu={}pe".format(intensity[2]))
plt.legend()
plt.savefig("./plots/offset_error.png",dpi=400)
plt.show()
