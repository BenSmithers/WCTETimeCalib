
"""
    Loads in the WCTE geometry with the mPMT _positions and PMT _positions 
"""

import numpy as np
import pandas as pd 
from math import pi, cos, sin 
from WCTECalib.utils import mm, set_axes_equal
import os 
N_CHAN = 19 # channels/PMTs per mPMT 

geo_file = os.path.join(os.path.dirname(__file__), "..","data", "geometry.csv")

if os.path.exists(geo_file):
    df = pd.read_csv(geo_file)
    _positions = [
        df["X"],
        df["Y"],
        df["Z"]
    ]
    N_MPMT=106
else:

    #N_MPMT = 200 # total number of mPMTs 

    # these are taken from Mohit's Geant4 mPMT code  
    _mpmt_xshift =  [0.*mm, 84.43*mm, 42.21*mm, -42.21*mm, -84.43*mm, -42.21*mm, 42.21*mm, 155.109*mm, 134.32*mm, 77.55*mm, 0.*mm, -77.55*mm, -134.32*mm, -155.109*mm, -134.32*mm, -77.55*mm, 0.*mm, 77.55*mm, 134.32*mm]
    _mpmt_yshift = [0.*mm, 0.*mm, -73.125*mm, -73.125*mm, 0.*mm, 73.125*mm, 73.125*mm, 0.*mm, -77.55*mm, -134.32*mm, -155.109*mm, -134.32*mm, -77.55*mm, 0.*mm, 77.55*mm, 134.32*mm, 155.109*mm, 134.32*mm, 77.55*mm]
    _mpmt_zshift= [0.*mm, -13.91*mm, -13.91*mm, -13.91*mm, -13.91*mm, -13.91*mm, -13.91*mm, -50.88*mm, -50.88*mm, -50.88*mm, -50.88*mm, -50.88*mm, -50.88*mm, -50.88*mm, -50.88*mm, -50.88*mm, -50.88*mm, -50.88*mm, -50.88*mm]


    _pmt_shifts = np.array([
        _mpmt_xshift, _mpmt_yshift, _mpmt_zshift
    ]).T 

    _positions = []
    _mPMT_id= []
    _channel = []

    """
        BUILD DEFAULT GEOMETRY 
    """

    for i in range(5):
        for j in range(5):
            # skip the caps 
            if (i==0 and j==0) or (i==0 and j==4) or (i==4 and j==0) or (i==4 and j==4):
                continue
            # get the next, unused mPMT number 
            if len(_mPMT_id)==0:
                this_mpmt= 0
            else:
                this_mpmt = _mPMT_id[-1]+1

            # we're filling in an endcap here, so this is just one of the locations on a grid at the bottom 
            step = 0.6 
            shift = np.array([
                0.6*(i -2),
                0.6*(j -2),
                0
            ])

            # add a relative shift for each mPMT 
            for chan in range(N_CHAN):
                this_pos = shift + _pmt_shifts
                _positions.append(this_pos[chan])
                _mPMT_id.append(this_mpmt)
                _channel.append(chan)

            # okay now we do the top cap
            # and so we build a rotation matrix for a 180deg rotation about the y-axis
            this_mpmt = this_mpmt + 1 
            rot_mat = np.array([
                [cos(pi),0,sin(pi) ],
                [0, 1, 0,],
                [-sin(pi), 0, cos(pi)]
            ])
            

            for chan in range(N_CHAN):
                # flip the shift vector according to the rotation (now the mPMT points down)
                rotated =  np.matmul(rot_mat, _pmt_shifts[chan])
                # apply the xy-plane shift, then the relative shift, and then the vertical shift (order irrelevant)
                this_pos = shift + rotated + np.array([0,0,3.4])

                _positions.append( this_pos )
                _mPMT_id.append(this_mpmt)
                _channel.append(chan)

    # four equally-spaced z-layers
    for z in [0.68, 1.36, 2.04, 2.72]:
        for angle in np.linspace(0, 2*pi, 16, endpoint=False):
            # rotate along y-axis to point inwards 
            rot_mat = np.array([
                [cos(-pi/2),0,sin(-pi/2) ],
                [0, 1, 0,],
                [-sin(-pi/2), 0, cos(-pi/2)]
            ])
            # rotate along z-axis to to align with the barrel correctly 
            rot_mat2 = np.array([
                [1, 0, 0 ],
                [0, cos(-angle), sin(-angle)],
                [0, -sin(-angle), cos(-angle)]
            ])

            # shifting to the outer barrel, and upwards by some amount 
            shift= np.array([
                cos(angle)*1.9,
                sin(angle)*1.9, 
                z
            ])

            full_rot = np.matmul( rot_mat, rot_mat2)
            this_mpmt = this_mpmt + 1 
            for chan in range(N_CHAN):
                rotated = np.matmul(full_rot, _pmt_shifts[chan])
                this_pos = shift + rotated 
                _positions.append(this_pos)
                _mPMT_id.append(this_mpmt)
                _channel.append(chan)

    N_MPMT = this_mpmt+1
    print(N_MPMT)

    _positions = np.transpose(_positions) 


    _data = np.transpose([
        _mPMT_id, 
        _channel, 
        _positions[0], 
        _positions[1], 
        _positions[2] 
    ])

    df = pd.DataFrame(
        columns = ["mPMT", "Chan", "X", "Y", "Z"],
        data = _data
    )

    df.to_csv(geo_file)


if __name__=="__main__":

    from mpl_toolkits import mplot3d

    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import axes3d
    fig = plt.figure()
    ax = plt.axes(projection="3d")
    ax.scatter(_positions[0], _positions[1], _positions[2])
    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.set_zlabel("Z [m]")
    set_axes_equal(ax)
    plt.show()

def get_pmt_positions():
    """
        Returns a numpy array of all the mPMT PMT _positions ()
    """
    return np.transpose([
        df["X"], 
        df["Y"],
        df["Z"]
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


