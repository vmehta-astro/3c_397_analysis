import yt
import glob
from yt.units import km
for file in glob.glob ("ddt_off_hdf5_plt_cnt_*"):
        ds = yt.load (file)
        slc = yt.SlicePlot (ds, "theta", "density", width=(2500 * km, 5000 * km))
        slc.pan_rel((-25.7144,0))
        slc.annotate_timestamp(corner='upper_right', draw_inset_box=True)
        slc.save()
