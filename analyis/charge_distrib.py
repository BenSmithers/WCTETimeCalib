import os 
import numpy as np 
from tqdm import tqdm 
import matplotlib.pyplot as plt 

"""
    Charge distribution plot
"""

datafile = os.path.join(
    os.path.dirname(__file__),
    "..",
    "data",
    "wcsim.npz"
)
data = np.load(datafile, allow_pickle=True) 

charges = data["digi_hit_charge"][:]


cbin = np.linspace(0, 10, 1001)
binned = np.zeros(len(cbin)-1)


for ir, runno in tqdm(enumerate(range(len(charges)))):
    these_charges = charges[runno]

    binned += np.histogram(these_charges, cbin)[0]

binned = binned/np.sum(binned)
plt.stairs(binned, cbin)
plt.grid(which='both', alpha=0.3)
plt.xlabel("digi-hit-charge [?]", size=14)
plt.ylabel("Arb. Units", size=14)
plt.xlim([0,5])
plt.show()
