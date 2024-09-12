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

data = np.loadtxt(filename, delimiter=' ', comments='#').T

_mpmt_xshift =  [0.*mm, 84.43*mm, 42.21*mm, -42.21*mm, -84.43*mm, -42.21*mm, 42.21*mm, 155.109*mm, 134.32*mm, 77.55*mm, 0.*mm, -77.55*mm, -134.32*mm, -155.109*mm, -134.32*mm, -77.55*mm, 0.*mm, 77.55*mm, 134.32*mm]
_mpmt_yshift = [0.*mm, 0.*mm, -73.125*mm, -73.125*mm, 0.*mm, 73.125*mm, 73.125*mm, 0.*mm, -77.55*mm, -134.32*mm, -155.109*mm, -134.32*mm, -77.55*mm, 0.*mm, 77.55*mm, 134.32*mm, 155.109*mm, 134.32*mm, 77.55*mm]
_mpmt_zshift= [0.*mm, -13.91*mm, -13.91*mm, -13.91*mm, -13.91*mm, -13.91*mm, -13.91*mm, -50.88*mm, -50.88*mm, -50.88*mm, -50.88*mm, -50.88*mm, -50.88*mm, -50.88*mm, -50.88*mm, -50.88*mm, -50.88*mm, -50.88*mm, -50.88*mm]
_pmt_shifts = np.array([
    _mpmt_xshift, _mpmt_yshift, _mpmt_zshift
])


_pmt_shifts = _pmt_shifts.T

xs = data[3]*mm
ys = data[4]*mm
zs = data[5]*mm



geo_file = os.path.join(os.path.dirname(__file__), "..","data", "geometry.csv")
_unique_id_count = 0


if os.path.exists(geo_file):
    df = pd.read_csv(geo_file)
    _positions = [
        df["X"],
        df["Y"],
        df["Z"]
    ]
    N_MPMT=106
else:
    channels = []
    _positions = [] 
    dirs = []
    mPMT_id= []
    led_mpmt = []
    unique_id = []
    led_pos = []
    led_dir = []
    led_ids = []
    for i in range(len(data[0])):
        shift = np.array([data[3][i], data[4][i], data[5][i]])*mm
        dx = data[6][i]
        dy = data[7][i]
        dz = data[8][i]

        rotation = data[9][i]*pi/180

        # rotate about y axis, angle should be based on direction in xz plane 
        theta = pi+np.arctan2(dy, dx)
        spinny = np.array([
                    [cos(rotation), sin(rotation), 0],
                    [-sin(rotation), cos(rotation), 0],
                    [0, 0, 1]
                ])
        
        if abs(dz)<1e-5:
            rot_mat = np.array([
                        [cos(-pi/2),0,sin(-pi/2)],
                        [0,1,0],
                        [-sin(-pi/2),0, cos(-pi/2)],
                    ])
            rot_mat2 = np.array([
                [1,0,0],
                [0, cos(-theta), sin(-theta)],
                [0, -sin(-theta), cos(-theta)]
            ])
            full_rot = np.matmul(np.matmul( rot_mat, rot_mat2), spinny)
            
        elif dz>0.5:
            full_rot = spinny
        else:

            rot_mat2 = np.array([
                        [1,0,0],
                        [0,cos(-pi),sin(-pi) ],
                        [0,-sin(-pi), cos(-pi)],
                    ])
            full_rot = np.matmul( spinny, rot_mat2)

        for chan in range(N_CHAN):
            rotated = np.matmul(full_rot, _pmt_shifts[chan])
            this_pos = shift + rotated 
            _positions.append(this_pos)
            channels.append(chan)
            mPMT_id.append(i)
            this_dir = this_pos - shift 
            dirs.append(this_dir / np.sqrt(np.sum(this_dir**2)))
            unique_id.append(_unique_id_count)
            _unique_id_count+=1
            
            if np.sum(this_dir**2)<1e-15:
                print(_unique_id_count)


    _positions = np.transpose(_positions)
    dirs = np.transpose(dirs)
    _data = np.transpose([
        unique_id, 
        mPMT_id, 
        channels, 
        _positions[0], 
        _positions[1],
        _positions[2], 
        dirs[0], 
        dirs[1],
        dirs[2]
    ])

    df = pd.DataFrame(
        columns = ["unique_id", "mPMT", "Chan", "X", "Y", "Z", "dx", "dy", "dz"],
        data = _data
    )

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