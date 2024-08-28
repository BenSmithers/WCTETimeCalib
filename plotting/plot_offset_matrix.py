
from WCTECalib.utils import convert_to_2d_offset

import matplotlib.pyplot as plt 
import pandas as pd
import numpy as np
import os 


if __name__=="__main__":

    fits = pd.read_csv(os.path.join(os.path.dirname(__file__), "..","data","calculated_offsets.csv"))
    matrix = convert_to_2d_offset(fits)

    plt.pcolormesh(range(len(matrix)), range(len(matrix)), matrix.T, vmin=-100, vmax=100, cmap="coolwarm")
    plt.xlabel("PMT ID")
    plt.ylabel("Relative to PMT ID")
    plt.ylim([0,100])
    plt.xlim([0,100])
    plt.colorbar()
    plt.show()
