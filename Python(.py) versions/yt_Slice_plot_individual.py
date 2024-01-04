import yt
import glob
from yt.units import km


cent = 1.25e8


for file in glob.glob ("ddt_off_hdf5_plt_cnt_*"):
    ds = yt.load(file)
    fields = [("gas","density"), ("gas","temperature"), ("gas", "pressure"), ("flash", "phfa")]
    #print(ds.field_list)
    slc = yt.SlicePlot(ds, 'theta', fields, origin = "native", center=[cent,0,0])
    #slc.set_width((20000*km, 20000*km))
    slc.set_cmap (field = "phfa" ,cmap = "hot")
    slc.set_zlim('phfa',0,1)
    slc.set_cmap (field = "density", cmap = "viridis")
    slc.set_zlim('density', 1e0, 1e10)
    slc.set_cmap (field = "temperature", cmap = "inferno")
    slc.set_zlim('temperature',1e7,1e10)
    slc.set_cmap (field = "pressure", cmap = "Spectral")
    slc.annotate_timestamp (corner='upper_right', draw_inset_box=True, redshift = False)
    slc.set_width((2*cent, 4*cent))
    #slc.annotate_grids()
    #slc.annotate_contour(('gas', 'temperature'))
    slc.save()
