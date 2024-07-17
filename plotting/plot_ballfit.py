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

sca = axes.scatter(range(n_diffs), np.zeros(n_diffs), ls='', marker='.', )

def animate(i): 
    #otimes = sample_balltime(ball= np.array([xs, ys, zs]))
    otimes = sample_balltime(ball= np.array([0.1,0.5 , 0.2]))
    fit = ballfit(otimes)

 
    distances = np.sqrt(np.sum((pmt_pos - fit[0:3])**2, axis=1))
    times = (1e9)*distances*N_WATER/C 

    offset_adjusted_t = otimes - np.array(offset_dict["calc_offset"]) - fit[3]
    diffs = times - offset_adjusted_t 
    #print("{} - {}".format(min(diffs), max(diffs)))

    def get_color(n, colormax=3.0, cmap="viridis"):
        """
            Discretize a colormap. Great getting nice colors for a series of trends on a single plot! 
        """
        this_cmap = plt.get_cmap(cmap)
        return this_cmap(n/colormax)

    ocolirs =get_color( (otimes-np.min(otimes))/(np.max(otimes)-np.min(otimes)), 1) 
    colors = get_color( (offset_adjusted_t-np.min(offset_adjusted_t))/(np.max(offset_adjusted_t)-np.min(offset_adjusted_t)), 1)

    if False:
        
        fig = plt.figure()
        ax = plt.axes(projection="3d")
        ax.scatter(offset_dict["X"], offset_dict["Y"], offset_dict["Z"], color=ocolirs)        
        ax.set_xlabel("X [m]")
        ax.set_ylabel("Y [m]")
        ax.set_zlabel("Z [m]")
        set_axes_equal(ax)
        plt.savefig("./plots/raw_times.png", dpi=400)
        plt.show()
        plt.clf()

        ax = plt.axes(projection="3d")
        ax.scatter(offset_dict["X"], offset_dict["Y"], offset_dict["Z"], color=colors)
        ax.plot(fit[0], fit[1], fit[2], 'rd')
        
        ax.set_xlabel("X [m]")
        ax.set_ylabel("Y [m]")
        ax.set_zlabel("Z [m]")
        set_axes_equal(ax)
        plt.savefig("./plots/ballfit.png", dpi=400)
        plt.show()
        plt.clf()
        

        # okay now go mPMT to mPMT 
        mPMT_IDs = range(N_MPMT)
        all_diffs = [] 
        """for id in mPMT_IDs:
            cut = offset_dict["mPMT"]==float(id)
            adjusted_diffs = diffs[cut]
            all_diffs+=adjusted_diffs.tolist()"""
        plt.clf()
        plt.plot(range(len(diffs)), offset_adjusted_t, ls='', marker='.')
        #plt.ylim([-0., 0.3])
        plt.xlabel("PMT No")
        plt.ylabel("Offset-Adjusted Times [ns]")
        plt.savefig("./plots/offset_adjusted.png", dpi=400)    
        plt.show()

    axes.clear()
    sizes = diffs**2
    colors = get_color(np.abs(diffs), 6, "inferno_r")
    axes.scatter(range(len(diffs)), diffs, sizes=sizes, c=colors)
    #axes.scatter(range(n_diffs), np.zeros(n_diffs), ls='', marker='.', )
    axes.set_ylim([-12, 12])
    axes.set_xlabel("PMT No")
    axes.set_ylabel("Diff from expected source-PMT time [ns]")
    #sca.set_offsets(np.array([range(len(diffs)), diffs]))
    return sca,

    #plt.savefig("./plots/time_deviation.png", dpi=400)
    
    #plt.show(block=False)
    #plt.pause(0.01)
import matplotlib.animation as animation

ani = animation.FuncAnimation(figs, animate, 100, repeat=True, interval=50)
writer = animation.PillowWriter(fps=15,
                                 metadata=dict(artist='Me'),
                                 bitrate=1800)
ani.save('./plots/scatter.gif', writer=writer)


