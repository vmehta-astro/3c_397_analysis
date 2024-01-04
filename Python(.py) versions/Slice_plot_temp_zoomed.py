import yt
import glob
from yt.units import km
for file in glob.glob ("ddt_off_hdf5_plt_cnt_*"):
        ds = yt.load (file)
        slc = yt.SlicePlot (ds,"theta", ("temperature"), width=(2500 * km, 5000 * km))
        slc.set_cmap(field=("temperature"), cmap="inferno")
        slc.annotate_timestamp(corner='upper_right', draw_inset_box=True)
        slc.pan_rel((-25.7144,0))           
        slc.save()




#To Zoom-in to WD use following equation to fit WD.
#sl.pan_rel((-x, 0) ------> x = (grid size/(2 * X width in km)) - 0.5 
