
import os 
import json 
import numpy as np 
from copy import deepcopy
from WCTECalib.utils import C, N_WATER, mm,ball_pos, second
from WCTECalib.alt_geo import df, N_CHAN, get_pmt_positions, N_MPMT

central_ball_loc = ball_pos

outfile = os.path.join(
    os.path.dirname(__file__),
    "..",
    "data",
    "wcsim_offset.json"
)

offsets = np.linspace(-155, 155, 3600)
binids = np.arange(-0.5, N_CHAN*N_MPMT+0.5, 1)
all_bins = np.zeros(( len(binids)-1, len(offsets)-1))
#simulation result
_obj = open(outfile, 'rt')
data = json.load(_obj)
_obj.close()

n_flash = len(data["times"])

for flash_id  in range(n_flash):
    _ids = np.array(data["pmtid"][flash_id])
    if 0 not in _ids:
        continue
    _t_meas= np.array(data["times"][flash_id])
    _charge = np.array(data["charge"][flash_id])

    these_data = np.array([_ids, _t_meas, _charge]).T 
    these_data = np.array(sorted(these_data, key=lambda x:x[1] ))

    ids = []
    charges = []
    t_meas = []
    for entry in these_data:
        if entry[0] not in ids:
            ids.append(entry[0])
            charges.append(entry[2])
            t_meas.append(entry[1])
    t_meas = np.array(t_meas)
    ids = np.array(ids)
    #charge = []

    # we need to filter this so only the earliest time entry is kept

    positions = get_pmt_positions(ids)

    distances = np.sqrt(np.sum( (positions - central_ball_loc)**2 , axis=1)) #predicted distances
    pred_time = second*distances*N_WATER/C
    
    calculated_offset = t_meas - pred_time 
    calculated_offset = calculated_offset[0]-calculated_offset 

    all_bins += np.histogram2d(ids, calculated_offset, bins=(binids, offsets))[0]

id_mesh, offset_mesh = np.meshgrid(0.5*(binids[1:] + binids[:-1]), 0.5*(offsets[1:] + offsets[:-1]))
# calculate the mean of the distribution
mean_offsets = np.sum(offset_mesh.T*all_bins, axis=1)/np.sum(all_bins, axis=1)

new_df = deepcopy(df)

new_df["calc_offset"] = mean_offsets - mean_offsets[0]

new_df.to_csv(
    os.path.join(os.path.dirname(__file__),"..", "data","calculated_offsets_lbmc.csv"),
    index=False
)
np.array(df["unique_id"]), mean_offsets
