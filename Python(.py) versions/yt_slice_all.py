import yt
import glob
import mpi4py
from yt.units import km
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import AxesGrid


cent = 3.1e9

yt.enable_parallelism()

cores = 1

files = glob.glob ("def_high_dens_hdf5_plt_cnt_000800")
files.sort()


for file in yt.parallel_objects(files, cores):
    
    fig = plt.figure()
    
    grid = AxesGrid(fig, (0.075, 0.075, 5, 1),
                    nrows_ncols = (1, 5), 
                    axes_pad= 2.0, 
                    label_mode="2", 
                    share_all = True, 
                    cbar_location = "right", 
                    cbar_mode = "each", 
                    cbar_size = "4%", 
                    cbar_pad="2%")
    
    ds = yt.load(file)
    fields = [("gas","density"), ("gas","temperature"), ("gas", "pressure"), ("flash", "phfa"), ("flash", "ye  ")]
    #print(ds.field_list)
    slc = yt.SlicePlot(ds, 'theta', fields, origin = "native", center=[cent,0,0])
    #slc.set_width((20000*km, 20000*km))
    slc.set_cmap (field = "phfa" ,cmap = "hot")
    slc.set_zlim('phfa',0.0e0,1.0e0)
    slc.set_log(('flash','phfa'), False)
    slc.set_cmap (field = "density", cmap = "viridis")
    slc.set_zlim('density', 1e0, 1e10)
    slc.set_cmap (field = "temperature", cmap = "inferno")
    slc.set_zlim('temperature',1e7,1e10)
    slc.set_cmap (field = "pressure", cmap = "Spectral")
    slc.set_zlim('pressure', 1.0e18, 1.0e28)
    slc.set_zlim('ye  ', 0.460,0.500)
    slc.set_cmap (field = "ye  ", cmap = "Spectral")
    slc.annotate_timestamp (corner='upper_right', draw_inset_box=True, redshift = False)
    slc.set_width((2*cent, 4*cent))
    #slc.annotate_grids()
    #slc.annotate_contour(('gas', 'temperature'))
#    slc.show()
    for i, field in enumerate(fields):
        plot = slc.plots[field]
        plot.figure = fig
        plot.axes = grid[i].axes
        plot.cax = grid.cbar_axes[i]
    slc._setup_plots()
    
    plt.savefig("./slice_plots/" + file + "_all_sliceplots.png" , bbox_inches="tight")
    ds.index.clear_all_data()
    #file.close()
