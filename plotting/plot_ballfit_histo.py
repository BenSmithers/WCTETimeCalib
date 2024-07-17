from WCTECalib.times import generate_offsets, sample_balltime
from WCTECalib.geometry_old import N_MPMT
from WCTECalib.utils import C, N_WATER, set_axes_equal

from WCTECalib.fitting import fit_ball as ballfit 
import numpy as np 
from math import pi 
import os 
import matplotlib.pyplot as plt 
import pandas as pd 

random_rsq = np.random.rand()*(1.9**2)
radii = np.sqrt(random_rsq)
random_angle = np.random.rand()*2*pi 

xs = np.cos(random_angle)*radii
ys = np.sin(random_angle)*radii
zs = np.random.rand()*3 

figs, axes = plt.subplots()
axes.set_ylim([-9,9])


offset_dict = pd.read_csv(
    os.path.join(os.path.dirname(__file__),
    "..",
    "data",
    "calculated_offsets.csv")
)
pmt_pos = np.transpose([
        offset_dict["X"], 
        offset_dict["Y"],
        offset_dict["Z"]
]) 

n_diffs = len(np.array(offset_dict["X"]))


def sample(): 
    #otimes = sample_balltime(ball= np.array([xs, ys, zs]))
    otimes = sample_balltime() #ball= np.array([0.1,0.5 , 0.2]))
    fit = ballfit(otimes)

 
    distances = np.sqrt(np.sum((pmt_pos - fit[0:3])**2, axis=1))
    times = (1e9)*distances*N_WATER/C 

    offset_adjusted_t = otimes - np.array(offset_dict["calc_offset"]) - fit[3]
    diffs = times - offset_adjusted_t 
    #print("{} - {}".format(min(diffs), max(diffs)))

    return diffs

n_samples = 10
many_samp = [sample() for i in range(n_samples)]

mean_err = np.mean(many_samp, axis=0)

def get_color(n, colormax=3.0, cmap="viridis"):
    """
        Discretize a colormap. Great getting nice colors for a series of trends on a single plot! 
    """
    this_cmap = plt.get_cmap(cmap)
    return this_cmap(n/colormax)


sizes = 0.75*mean_err**2 + 0.2

twosig = np.percentile(np.abs(mean_err), 68.2689492)
threesig = np.percentile(np.abs(mean_err), 99.7300204)

colors = get_color(np.abs(mean_err), 6, "RdBu")
plt.fill_between([0, len(mean_err)], -threesig, threesig, label=r"3$\sigma$", color="green")

plt.fill_between([0, len(mean_err)], -twosig, twosig, label=r"2$\sigma$",color="yellow")
plt.scatter(range(len(mean_err)), mean_err,  sizes=sizes, c='k')
plt.xlabel("PMT No", size=14)
plt.ylabel("Mean Error", size=14)
plt.title("{} Samples".format(n_samples),size=14)
plt.ylim([-11,11])
plt.legend()
plt.savefig("./plots/regions_of_err.png")
plt.show()