import yt
import glob
import mpi4py
from yt.units import km
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import AxesGrid
import numpy as np


cent = 2.0e9
yt.enable_parallelism()

cores = 16

files = glob.glob ("def_high_dens_hdf5_chk_*")
files.sort()


for file in yt.parallel_objects(files, cores):
    fig = plt.figure()

    grid = AxesGrid(fig, (0.075, 0.075, 2, 1),
                    nrows_ncols = (1, 2),
                    axes_pad= 2.0,
                    label_mode="2",
                    share_all = True,
                    cbar_location = "right",
                    cbar_mode = "each",
                    cbar_size = "4%",
                    cbar_pad="2%")

    ds = yt.load(file)
    ad = ds.all_data()
    #for i in sorted(ds.field_list):
    #    print(i)

    def binding_energy(field, data):
        return(data["gas", "density"] * ((0.5 * ((data[("flash", "velx")])**2 + (data[("flash", "vely")])**2)) + data[("flash", "eint")] + data[("flash", "gpot")]))

    ds.add_field(("flash", "bneg"), function = binding_energy, sampling_type = "cell", units = "auto")# , dimensions = dimensions.energy)

    #print(ds.field_list)
    positive_binding_energy = ad.cut_region(['obj["flash", "bneg"] > 1.0e17'])
    negative_binding_energy = ad.cut_region(['obj["flash", "bneg"] < -1.0e17'])
    pos_bn_eg_without_fluff = positive_binding_energy.cut_region(['obj["gas", "density"] > 1.0e-2'])
    neg_bn_eg_wihtout_fluff = negative_binding_energy.cut_region(['obj["gas", "density"] > 1.0e-2'])

    #print("%.6E" %len(ad["flash", "bneg"]))
    #print("%.6E" %len(positive_binding_energy["flash", "bneg"]))
    #print("%.6E" %len(negative_binding_energy["flash", "bneg"]))

    slc1 = yt.SlicePlot(ds, 'theta', ("flash","bneg"), center=[cent,0,0], data_source = pos_bn_eg_without_fluff)
    slc1.set_zlim(("flash", "bneg"), 1.0e18, 1.0e22)
    slc1.annotate_title("Specific Binding Energy Density (unbouned)")
    slc1.set_cmap (field = "bneg", cmap = "plasma")
    slc1.annotate_timestamp (corner='upper_right', draw_inset_box=True, redshift = False)
    slc1.set_width((2*cent, 4*cent))
    #slc1.save("binding_energy_plot(unbounded).png")

    slc2 = yt.SlicePlot(ds, 'theta', ("flash","bneg"), center=[cent,0,0], data_source = neg_bn_eg_wihtout_fluff)
    slc2.set_zlim(("flash", "bneg"), -1.0e20, -1.0e24)
    slc2.set_log(("flash", "bneg"), linthresh = 1.0e20)
    slc2.annotate_title("Specific Binding Energy Density (bounded)")
    slc2.set_cmap (field = "bneg", cmap = "plasma")
    slc2.annotate_timestamp (corner='upper_right', draw_inset_box=True, redshift = False)
    slc2.set_width((2*cent, 4*cent))
    #slc2.save("binding_energy_plot(bounded).png")

    #for i, field in enumerate(fields):
    plot1 = slc1.plots[("flash","bneg")]
    plot2 = slc2.plots[("flash","bneg")]

    plot1.figure = fig
    plot2.figure = fig
    
    plot1.axes = grid[1].axes
    plot2.axes = grid[0].axes
    
    plot1.cax = grid.cbar_axes[1]
    plot2.cax = grid.cbar_axes[0]

    slc1._setup_plots()
    slc2._setup_plots()


    plt.savefig("./slice_plots/" + file + "_specific_binding_energy_plots.png" , bbox_inches="tight")
    ds.index.clear_all_data()
