import matplotlib.pyplot as plt 

import numpy as np 
from WCTECalib.utils import set_axes_equal, get_color
from math import cos, sin, pi
import os 
import pandas as pd 
N_CHAN =19 
filename = os.path.join(
    os.path.dirname(__file__),
    "..",
    "data",
    "mPMT_Position_WCTE.txt"
)

mm = 1e-3
m = 1



geo_file = os.path.join(os.path.dirname(__file__), "..","data", "geometry.csv")


_unique_id_count = 0


if os.path.exists(geo_file):
    df = pd.read_csv(geo_file)
    _positions = [
        df["X"],
        df["Y"],
        df["Z"]
    ]
    channels= df["Chan"]
    N_MPMT=97
else:
    mc_geo_file = os.path.join(os.path.dirname(__file__), "..","data", "geodump.npz")
    mc_geo = np.load(mc_geo_file, allow_pickle=True)

    unique_id = mc_geo["tube_no"]
    _positions = mc_geo["position"]/100
    dirs = mc_geo["orientation"]
    N_MPMT= int(len(unique_id)/N_CHAN)
    print(N_MPMT)
    channels = list(range(N_CHAN))*N_MPMT
    assert len(channels) == len(unique_id)
    mPMT_id = np.repeat(range(N_CHAN), N_MPMT)


    led_mpmt = []
    led_pos = []
    led_dir = []
    led_ids = []
    


    _positions = np.transpose(_positions)
    dirs = np.transpose(dirs)
    _data = np.transpose([
        unique_id, 
        mPMT_id, 
        channels, 
        _positions[0], 
        _positions[1],
        -1*_positions[2], 
        dirs[0], 
        dirs[1],
        dirs[2]
    ])

    df = pd.DataFrame(
        columns = ["unique_id", "mPMT", "Chan", "X", "Y", "Z", "dx", "dy", "dz"],
        data = _data
    )  
    df.to_csv(geo_file, index=False)

    channels =np.array(channels)
    _positions = np.array(_positions)

if __name__=="__main__":
    from mpl_toolkits import mplot3d

    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import axes3d
    colors = get_color(channels, 19, "inferno")


    fig = plt.figure()
    ax = plt.axes(projection="3d")
    ax.scatter(_positions[0], _positions[1], _positions[2] , color=colors)
    ax.set_xlabel("X [m]")#
    ax.set_ylabel("Y [m]")
    ax.set_zlabel("Z [m]")
    set_axes_equal(ax)
    plt.show()



def get_pmt_positions(ids=None):
    """
        Returns a numpy array of all the mPMT PMT _positions ()
    """
    if ids is None:
        return np.transpose([
            df["X"], 
            df["Y"],
            df["Z"]
        ]) 
    else:
        return np.transpose([
            df["X"][df["unique_id"].isin(ids)], 
            df["Y"][df["unique_id"].isin(ids)],
            df["Z"][df["unique_id"].isin(ids)]
        ]) 
    
def get_pmt_dirs(ids = None):
    if ids is None:
        return np.transpose([
            df["dx"], 
            df["dy"],
            df["dz"]
        ]) 
    else:
        return np.transpose([
            df["dx"][df["unique_id"].isin(ids)], 
            df["dy"][df["unique_id"].isin(ids)],
            df["dz"][df["unique_id"].isin(ids)]
        ]) 


def get_mPMT_pos(_mPMT_id:int):
     return np.transpose([
        df[df["mPMT"]==_mPMT_id]["X"], 
        df[df["mPMT"]==_mPMT_id]["Y"], 
        df[df["mPMT"]==_mPMT_id]["Z"], 
    ])


def get_pmt_pos_by_channel(_channel:int):
    return np.transpose([
        df[df["Chan"]==_channel]["X"], 
        df[df["Chan"]==_channel]["Y"], 
        df[df["Chan"]==_channel]["Z"], 
    ])

def get_pmt_pos(_mPMT_id: int, _channel:int):
    filtered = df.query('Chan=={} & mPMT=={}'.format(_channel, _mPMT_id))
    return np.array(
        filtered["X"], 
        filtered["Y"], 
        filtered["Z"]
    )