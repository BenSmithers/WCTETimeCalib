
"""
    Loads in the WCTE geometry with the mPMT _positions and PMT _positions 
"""

import numpy as np
import pandas as pd 
from math import pi 
import os 
from WCTECalib.utils import set_axes_equal, C, N_WATER
import json 
from Geometry.WCD import WCD
N_CHAN = 19
PMT_MAX = 85*pi/180
LED_MAX = 25 #*pi/180

wcte = WCD("wcte", kind="WCTE")

_obj = open(os.path.join(os.path.dirname(__file__),"geodata", "modulo_dump.json"), 'rt')
geodata = json.load(_obj)['data']
_obj.close()

geo_file = os.path.join(os.path.dirname(__file__), "..","data", "geometry.csv")
led_file = os.path.join(os.path.dirname(__file__), "..","data", "leds.csv")
def fix_cord(vec):
    new_vec = np.zeros_like(vec)    
    new_vec[0] = vec[0]
    new_vec[1] = vec[1]
    new_vec[2] = vec[2]
    return new_vec

if os.path.exists(geo_file) and os.path.exists(led_file):
    df = pd.read_csv(geo_file)
    _positions = [
        df["X"],
        df["Y"],
        df["Z"]
    ]
    N_MPMT=106

    led_data = pd.read_csv(led_file)
    _led_pos = [
        led_data["X"],
        led_data["Y"],
        led_data["Z"]
    ]
    _led_dir = [
        led_data["dx"],
        led_data["dy"],
        led_data["dz"]
    ]


else:
    _unique_id_count = 0
    led_id = 0

    _positions = []
    _dirs = []
    _mPMT_id= []
    _led_mpmt = []
    _channel = []
    _unique_id = []
    _led_pos = []
    _led_dir = []
    _led_ids = []
    _feeds = []

    for im, mpmt in enumerate(wcte.mpmts):


        # let's make the mpmtID/channel map 
        if geodata[im]["MPMTIN"] is None:
            continue
        
        loc = mpmt.get_xy_points('design','feedthrough', wcte)
        _feeds.append(loc)

        # this does pmt->channel, we want the opposite 
        pmt_to_chan = geodata[im]["channel_mapping"]
        chan_to_pmt = {}
        for pmt_number in range(19):
            channel_number = int(pmt_to_chan["pmt_"+str(pmt_number)+"_chan_id"])
            chan_to_pmt[channel_number]=pmt_number

        for iled in range(3):
            if len( mpmt.leds)==0:
                continue
            this_led = mpmt.leds[iled].get_placement('design', wcte)
            _led_pos.append(fix_cord(this_led["location"])/1000.)
            _led_dir.append(fix_cord(this_led["direction_z"]))
            _led_ids.append(led_id)
            _led_mpmt.append(im)
            led_id += 1
        
        for ip, pmt in enumerate(mpmt.pmts):
            pdata = pmt.get_placement('design', wcte)

            _positions.append(fix_cord(pdata["location"])/1000.)
            _dirs.append(fix_cord(pdata["direction_z"]))
            _mPMT_id.append(im)
            _channel.append(ip)
            _unique_id.append(im*19 + ip)

    _positions = np.transpose(_positions) 
    _dirs = np.transpose(_dirs)
    _led_pos = np.transpose(_led_pos)
    _led_dir = np.transpose(_led_dir)
    _feeds = np.transpose(_feeds)/1000
    N_MPMT = im+1

    _data = np.transpose([
        _unique_id, 
        _mPMT_id, 
        _channel, 
        _positions[0], 
        _positions[1],
        _positions[2], 
        _dirs[0], 
        _dirs[1],
        _dirs[2]
    ])



    df = pd.DataFrame(
        columns = ["unique_id", "mPMT_slot_id", "Chan", "X", "Y", "Z", "dx", "dy", "dz"],
        data = _data
    )

    df.to_csv(geo_file, index=False)


    _data = np.transpose([
        _led_ids, 
        _led_mpmt, 
        _led_pos[0], 
        _led_pos[1],
        _led_pos[2], 
        _led_dir[0], 
        _led_dir[1],
        _led_dir[2]
    ])
    led_data = pd.DataFrame(
        columns = ["unique_id", "mPMT", "X", "Y", "Z", "dx", "dy", "dz"],
        data = _data
    )
    led_data.to_csv(led_file, index=False)

if __name__=="__main__":

    from mpl_toolkits import mplot3d

    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import axes3d
    from WCTECalib.utils import get_color
    fig = plt.figure()
    ax = plt.axes(projection="3d")
    cut = _positions[1]>-1000 
    #fcut = _feeds[1]>-10
    colors = get_color(df["Chan"], 19, "inferno")
    ax.scatter(_positions[0][cut], _positions[1][cut], _positions[2][cut], color=colors)
    #ax.scatter(_feeds[0][fcut], _feeds[1][fcut], _feeds[2][fcut],'green')
    #ax.quiver(_led_pos[0], _led_pos[1], _led_pos[2],_led_dir[0], _led_dir[1], _led_dir[2],color='red', length=0.1, normalize=True)
    ax.set_xlabel("X [m]")#
    ax.set_ylabel("Y [m]")
    ax.set_zlabel("Z [m]")
    set_axes_equal(ax)
    plt.show()



def get_pmts_visible(led_id:int, led_max_angle=LED_MAX):
    """
        Returns a filter (an array of bools) for which PMTs are visible for an LED flash 

        the "max angle" is half of the maximum opening angle the PMT emits light
    """

    led_max_angle_rad = led_max_angle*pi/180 

    led_ar_pos = get_led_positions([led_id,])
    led_pos = led_ar_pos[0]
    led_dir = get_led_dirs([led_id,])[0]

    pmt_pos = get_pmt_positions()
    pmt_dir = get_pmt_dirs()


    dvec = led_pos - pmt_pos

    pmt_light_angle = np.arccos(np.sum(pmt_dir*dvec, axis=1) / np.sqrt(np.sum(dvec**2, axis=1)))
    led_light_angle = np.arccos(np.sum(led_dir*(-1*dvec), axis=1) / np.sqrt(np.sum(dvec**2, axis=1)))


    keep = np.logical_and(pmt_light_angle<PMT_MAX, led_light_angle < led_max_angle_rad)
 
    return keep#  df[keep]["unique_id"]


def get_led_positions(ids=None):
    """
        Returns a numpy array of all the mPMT PMT _positions ()
    """
    if ids is None:
        return np.transpose([
            led_data["X"], 
            led_data["Y"],
            led_data["Z"]
        ]) 
    else:
        return np.transpose([
            led_data["X"][led_data["unique_id"].isin(ids)], 
            led_data["Y"][led_data["unique_id"].isin(ids)],
            led_data["Z"][led_data["unique_id"].isin(ids)]
        ]) 
    
def get_led_dirs(ids = None):
    if ids is None:
        return np.transpose([
            led_data["dx"], 
            led_data["dy"],
            led_data["dz"]
        ]) 
    else:
        return np.transpose([
            led_data["dx"][df["unique_id"].isin(ids)], 
            led_data["dy"][df["unique_id"].isin(ids)],
            led_data["dz"][df["unique_id"].isin(ids)]
        ]) 


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
