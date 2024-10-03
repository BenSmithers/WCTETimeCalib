from mpl_toolkits import mplot3d
from WCTECalib.utils import set_axes_equal, get_color
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
import pandas as pd 
import numpy as np 
import os 

ERRORS = True 


true = pd.read_csv(os.path.join(os.path.dirname(__file__), "..","data","offsets.csv"))

labels = ["Lock", "No Lock"]
offset_names = ["calculated_offsets_lock.csv", "calculated_offsets_lbmc.csv" ]

fig2d, axes = plt.subplots(1,1)
for i, name in enumerate(offset_names):

    offsets = pd.read_csv(os.path.join(os.path.dirname(__file__), "..","data",name))

    times = offsets["calc_offset"]
    true_offsets = true["offsets"]
    ids = true["unique_id"]



    if ERRORS:
        times = times - (true_offsets[0] -true_offsets)
        #times = times - true_offsets
        bad = times<-1.0
        print(list(ids[bad]))

    print("{} - {}".format(min(times), max(times))) 
    if ERRORS:
        colors =  (times+1)/2 - 0.5 # get_color( (times+1)/2, 1)
    else:
        colors = (times -min(times))/(max(times)-min(times)) # get_color( (times -min(times))/(max(times)-min(times)), 1)

    if False:
        fig = plt.figure()
        ax = plt.axes(projection="3d")
        cut = times<-0.5
        #ax.pcolormesh([0, 1], [0,1], [[0,]], vmin=-0.5, vmax=0.5, cmap="coolwarm", )
        scatterpts = ax.scatter(offsets["X"][cut], offsets["Y"][cut], offsets["Z"][cut], c=times[cut], cmap="coolwarm", vmin=-2,vmax=2)
        ax.set_xlabel("X [m]")
        ax.set_ylabel("Y [m]")
        ax.set_zlabel("Z [m]")
        cbar = fig.colorbar(scatterpts)
        cbar.set_label("Error [ns]",size=14)
        set_axes_equal(ax)
        plt.savefig("./plots/offsets_corrected.png", dpi=400)
        plt.show()


    if ERRORS:
        bins = np.linspace(-2,2, 200)
        histo = np.histogram(times, bins=bins)[0]
        plt.text(x=2.4, y=65+i*30, s=labels[i]+": ")
        plt.text(x=2.5, y=50+i*30, s="{:.2f} +/- {:.2f}".format(np.mean(times), np.std(times)))
            
        axes.stairs(histo, bins, label=labels[i])

axes.set_xlabel("Offset Error [ns]", size=14)
axes.set_ylabel("Counts")
fig2d.legend()
fig2d.tight_layout()

fig2d.savefig("./plots/offsets_error.png",dpi=400)
plt.show()

