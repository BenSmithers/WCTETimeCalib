"""
    Uses some toy MC to calculate an approximate vertex error based on the noise in the mPMT 

    for a given noise scale, we generate a series of possible offisets in the mPMT mainboards
    Then we fit to some sample pulses
    and generate a pseudo-flasher ball simulation from some random place
    and we see how accurate the fit ball location is using the offsets we fit 
    
"""

from WCTECalib.times import generate_offsets, sample_balltime, DIFFUSER_ERR, BALL_ERR, mm, ns
from WCTECalib.fitting import fit_ball as ballfit 
from calculate_shifts import refit 
import numpy as np 
from math import pi 
from tqdm import tqdm 
import os 
import matplotlib.pyplot as plt 

noises = np.logspace(-1, 0.5, 6 )
SAMPLES = 200


diff_errs=  [0, DIFFUSER_ERR, 10*ns]

ball_errs = [0, BALL_ERR, 10*mm]

def get_color(n, colormax=3.0, cmap="viridis"):
    """
        Discretize a colormap. Great getting nice colors for a series of trends on a single plot! 
    """
    this_cmap = plt.get_cmap(cmap)
    return this_cmap(n/colormax)

for ib, ball_err in enumerate(ball_errs):
    color = get_color(ib+1,4 )
    

    for id, diff_error in enumerate(diff_errs):
        ls = ["-", "--", '-.'][id]
        diffs = []

        for i, noise in enumerate(noises):
            
            random_rsq = np.random.rand(SAMPLES)*(1.9**2)
            radii = np.sqrt(random_rsq)
            random_angle = np.random.rand(SAMPLES)*2*pi 

            xs = np.cos(random_angle)*radii
            ys = np.sin(random_angle)*radii
            zs = np.random.rand(SAMPLES)*3 
            print("next noise")

            these_diffs = []
            for i in tqdm(range(SAMPLES)):
                
                generate_offsets()

                # fitting with a biased ball position
                ballpos = refit(noise, ball_err) 
                times = sample_balltime(noise, ball=np.array([xs[i], ys[i], zs[i]]), ball_pos_noise=False, diff_err=diff_error)

                fit = ballfit(times)

                these_diffs.append(
                    np.sqrt(sum([(fit[0] - xs[i])**2, (fit[1]-ys[i])**2, (fit[2] - zs[i])**2]))
                )
            diffs.append(these_diffs)
            label=None #"Ball Err {:.1f}mm; Diff err {:.1f}ns".format(ball_err*1000, diff_error)

        print(len(noises), len(np.mean(diffs, axis=1)))
        plt.errorbar(
            noises,
            np.mean(diffs, axis=1),
            yerr=np.std(diffs, axis=1),
            capsize=5, 
            marker='d',
            ls=ls,
            color=color,
            markerfacecolor=color,
            ecolor='k',
            label=label
            )
plt.plot([],[],ls="-", color='k',label="Diff err 0.0ns")
plt.plot([],[],ls="--", color='k',label="Diff err 1.0ns")
plt.plot([],[],ls="-.", color='k',label="Diff err 10.0ns")
plt.plot([],[],color=get_color(0+1,4 ), label="Ball Err 0.0mm")
plt.plot([],[],color=get_color(1+1,4 ), label="Ball Err 2.0mm")
plt.plot([],[],color=get_color(2+1,4 ), label="Ball Err 10.0mm")
plt.xlim([0, 3.3])
plt.xlabel("PMT TTS [ns]", size=14)
plt.legend()
plt.ylabel("Vertex Err [m]", size=14)        
plt.savefig(os.path.join(os.path.dirname(__file__), "plots","vertex_error.png"), dpi=400)
plt.show()