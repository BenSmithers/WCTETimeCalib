from WCTECalib.times import generate_offsets, sample_balltime
from WCTECalib.geometry import N_MPMT, get_pmt_positions
from WCTECalib.utils import C, N_WATER, set_axes_equal

from WCTECalib.fitting import fit_hits 
import numpy as np 
from math import pi 
import os 
import matplotlib.pyplot as plt 
import pandas as pd 

random_rsq = np.random.rand()*(1.2**2)
radii = np.sqrt(random_rsq)
random_angle = np.random.rand()*2*pi 

offset_dict = pd.read_csv(
    os.path.join(os.path.dirname(__file__),
    "..",
    "data",
    "calculated_offsets.csv")
)

n_diffs = len(np.array(offset_dict["X"]))

#sca = axes.scatter(range(n_diffs), np.zeros(n_diffs), ls='', marker='.', )

def main(): 

    xs = np.cos(random_angle)*radii
    zs = np.sin(random_angle)*radii
    ys = np.random.rand()*2-0.5

    ids, otimes, mus = sample_balltime(ball= np.array([xs, ys, zs]), mu=1)
    fit = fit_hits(ids, otimes)
    pmt_pos = get_pmt_positions(ids)

 
    distances = np.sqrt(np.sum((pmt_pos - fit[0:3])**2, axis=1))
    times = (1e9)*distances*N_WATER/C 

    offset_adjusted_t = otimes + np.array(offset_dict["calc_offset"])[ids] - fit[3]

    diffs = times - offset_adjusted_t 

    def get_color(n, colormax=3.0, cmap="viridis"):
        """
            Discretize a colormap. Great getting nice colors for a series of trends on a single plot! 
        """
        this_cmap = plt.get_cmap(cmap)
        return this_cmap(n/colormax)

    ocolirs =get_color( (otimes-np.min(otimes))/(np.max(otimes)-np.min(otimes)), 1) 
    colors = get_color( (offset_adjusted_t-np.min(offset_adjusted_t))/(np.max(offset_adjusted_t)-np.min(offset_adjusted_t)), 1)

    if False:
        ax = plt.axes(projection="3d")
        ax.scatter(offset_dict["X"], offset_dict["Y"], offset_dict["Z"], color=ocolirs)        
        ax.set_xlabel("X [m]")
        ax.set_ylabel("Y [m]")
        ax.set_zlabel("Z [m]")
        set_axes_equal(ax)
        plt.savefig("./plots/raw_times.png", dpi=400)
        plt.show()
        plt.clf()

        plt.plot(range(len(diffs)), offset_adjusted_t, ls='', marker='.')
        #plt.ylim([-0., 0.3])
        plt.xlabel("PMT No")
        plt.ylabel("Offset-Adjusted Times [ns]")
        plt.savefig("./plots/offset_adjusted.png", dpi=400)    
        plt.show()

    error  = np.sqrt( (xs - fit[0])**2 + (ys-fit[1])**2 + (zs-fit[2])**2)


    ax = plt.axes(projection="3d")
    ax.scatter(pmt_pos.T[0], pmt_pos.T[1], pmt_pos.T[2], color=colors)
    ax.plot(xs, ys, zs, 'bo')
    ax.plot(fit[0], fit[1], fit[2], 'rd')
    
    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.set_zlabel("Z [m]")
    set_axes_equal(ax)
    plt.savefig("./plots/ballfit.png", dpi=400)
    plt.show()
    plt.clf()
    
    return error

    

if __name__=="__main__":
    error = main()
