from mpl_toolkits import mplot3d
from WCTECalib.utils import set_axes_equal, get_color
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
import pandas as pd 
import numpy as np 
import os 

ERRORS = False 


true = pd.read_csv(os.path.join(os.path.dirname(__file__), "..","data","offsets.csv"))

labels = ["Lock" ]# , "Free"]
#offset_names = ["calculated_offsets_lock.csv" , "calculated_offsets_lbmc.csv" ]
offset_names = ["calculated_offsets_realdata.csv", ]
#offset_names = [ "calculated_offsets_lbmc.csv" ]
badbatch = np.array([57, 58, 81])

fig2d, axes = plt.subplots(1,1)
for i, name in enumerate(offset_names):

    offsets = pd.read_csv(os.path.join(os.path.dirname(__file__), "..","data",name))

    times = offsets["calc_offset"]
    errr = offsets["offset_sigma"]
    nhits = offsets["nhits"]
    #true_offsets = true["offsets"]
    ids = offsets["unique_id"]

    mPMT = ids // 19 
    
    wonky = np.array([each in badbatch for each in ids])
    times[wonky] = np.nan

    print("{} - {}".format(min(times), max(times))) 
    if ERRORS:
        colors =  (times+1)/2 - 0.5 # get_color( (times+1)/2, 1)
    else:
        colors = (times -min(times))/(max(times)-min(times)) # get_color( (times -min(times))/(max(times)-min(times)), 1)

    if True:
        fig = plt.figure()
        ax = plt.axes(projection="3d")
        cut = np.logical_not(np.isnan(times))

        print(len(offsets["X"][cut]))    

        #ax.pcolormesh([0, 1], [0,1], [[0,]], vmin=-0.5, vmax=0.5, cmap="coolwarm", )
        scatterpts = ax.scatter(offsets["X"][cut], offsets["Y"][cut], offsets["Z"][cut], c=times[cut], cmap="RdBu", vmin=np.nanmin(times), vmax=np.nanmax(times))
        ax.set_xlabel("X [m]")
        ax.set_ylabel("Y [m]")
        #ax.set_title("{}".format(labels[i]))
        ax.set_zlabel("Z [m]")
        cbar = fig.colorbar(scatterpts)
        #cbar.set_label("Distribution Width [ns]",size=14)
        set_axes_equal(ax)
        plt.savefig("./plots/offsets_corrected.png", dpi=400)

        fig = plt.figure()
        ax = plt.axes(projection="3d")
        cut = np.logical_not(np.isnan(times))
        #ax.pcolormesh([0, 1], [0,1], [[0,]], vmin=-0.5, vmax=0.5, cmap="coolwarm", )
        scatterpts = ax.scatter(offsets["X"][cut], offsets["Y"][cut], offsets["Z"][cut], c=errr[cut], cmap="inferno", vmin=0, vmax=2.5)
        ax.set_xlabel("X [m]")
        ax.set_ylabel("Y [m]")
        #ax.set_title("{}".format(labels[i]))
        ax.set_zlabel("Z [m]")
        cbar = fig.colorbar(scatterpts)
        #cbar.set_label("Distribution Width [ns]",size=14)
        set_axes_equal(ax)
        plt.savefig("./plots/offsets_corrected.png", dpi=400)

        if True:
            fig = plt.figure()
            ax = plt.axes(projection="3d")
            cut = np.logical_not(np.isnan(times))
            #ax.pcolormesh([0, 1], [0,1], [[0,]], vmin=-0.5, vmax=0.5, cmap="coolwarm", )
            scatterpts = ax.scatter(offsets["X"][cut], offsets["Y"][cut], offsets["Z"][cut], c=nhits[cut], cmap="inferno", vmin=0, vmax=np.max(nhits))
            ax.set_xlabel("X [m]")
            ax.set_ylabel("Y [m]")
            #ax.set_title("{}".format(labels[i]))
            ax.set_zlabel("Z [m]")
            cbar = fig.colorbar(scatterpts)
            #cbar.set_label("Distribution Width [ns]",size=14)
            set_axes_equal(ax)
            plt.savefig("./plots/offsets_amplitude.png", dpi=400)
            #plt.show()

            ybin = np.linspace(-2, 2, 20)
            hitbin = np.linspace(0,2000, 21)
            fig = plt.figure()
            histo = np.histogram2d(offsets["Y"][cut], nhits[cut], bins=(ybin, hitbin))[0]
            plt.pcolormesh(ybin, hitbin, histo.T, cmap="inferno")
            plt.xlabel("Y [m]", size=14)
            plt.ylabel("Amplitude [hits]", size=14)
            cbar = plt.colorbar()
            cbar.set_label("n PMTs")
            plt.savefig("./plots/z_vs_hits.png", dpi=400)



    if ERRORS:
        bins = np.linspace(-2,2, 200)
        histo = np.histogram(times, bins=bins)[0]
        axes.text(x=2.4, y=65+i*30, s=labels[i]+": ")
        axes.text(x=2.5, y=50+i*30, s="{:.2f} +/- {:.2f}".format(np.mean(times), np.std(times)))
            
        axes.stairs(histo, bins, label=labels[i])

axes.set_xlabel("Offset Error [ns]", size=14)
axes.set_ylabel("Counts")
fig2d.legend()
fig2d.tight_layout()

fig2d.savefig("./plots/offsets_error.png",dpi=400)
plt.show()

