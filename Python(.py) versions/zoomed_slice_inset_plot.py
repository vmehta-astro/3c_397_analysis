##################################################################################################
#Author - Vrutant Mehta (2022)
#This is a script for SlicePlots with inset plot with zoom-in to 
#the perticular area of interest. Make changes as per your requiremnts. 
##################################################################################################

#define required libraries and modules for the plots
import yt
import glob
from yt.units import km
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset


#constants for length and widht of the actual plots in CGS units.
cent = 3.0e9
cent2 = 1.25e8

#uncomment these lines for parallelization
#----
yt.enable_parallelism()
cores = 1

files = glob.glob ("def_high_dens_hdf5_plt_cnt_000800")
files.sort()


for file in yt.parallel_objects(files, cores):
#-----
#for loop for multiple files
#in case of parallelization, comment this for loop initilization and use line above this

#for file in glob.glob ("ddt_off_hdf5_plt_cnt_000000"): #define name of your file or files here

    fig, ax = plt.subplots(1,2) #begin the axes plots with subplots
    
    #making first main image
    ds = yt.load(file)
    fields = [("gas","temperature")]
    slc = yt.SlicePlot(ds, 'theta', fields, origin = "native", center=[2*cent,0,0])
    slc.set_cmap (field = "temperature", cmap = "inferno")
    slc.set_zlim('temperature',1e7,1e10)
    slc.show_colorbar(field = [("gas","temperature")])
    slc.annotate_timestamp (corner='upper_right', draw_inset_box=True, redshift = False)
    slc.set_width((4*cent, 4*cent))
    
    #functions to redraw this sliceplots on subplot of axes
    temp_plot = slc.plots[("gas","temperature")]
    temp_plot.figure = fig
    temp_plot.axes = ax[0].axes
    temp_plot.cax = fig.add_axes(ax[1])
    
    #adding inset plot into original plot and add some modification(edgecolor,labels, X-lim and Y-lim, etc.)
    inner_plot = inset_axes(ax[0].axes, width = "35%", height= "70%", loc="right",borderpad = 5)
    inner_plot.spines['bottom'].set_color('white')
    inner_plot.spines['top'].set_color('white')
    inner_plot.spines['left'].set_color('white')
    inner_plot.spines['right'].set_color('white')
    inner_plot.xaxis.label.set_color('white')
    inner_plot.yaxis.label.set_color('white')
    inner_plot.tick_params(axis='x', which='both', colors='white')
    inner_plot.tick_params(axis='y', which='both', colors='white')
    inner_plot.set_xlim(0,2*cent2)
    inner_plot.set_ylim(-2*cent2,2*cent2)
    
    #redraw zoom-in version of the sliceplot
    i = yt.SlicePlot(ds, 'theta', fields, origin = "native" ,center=[cent2,0,0])
    i.set_cmap (field = "temperature", cmap = "inferno")
    i.set_zlim('temperature',1e7,1e10)
    i.set_width((2*cent2, 4*cent2))
    
    #function to redraw this zoom-in version on the inset plot
    mini_plot = i.plots[("gas","temperature")]
    mini_plot.axes = inner_plot.axes
    
    #finally, setup everything and redraw whole figure again.
    slc._setup_plots()
    i._setup_plots()
    
    #marking the inset plot with region of the actual plot
    mark_inset(ax[0], inner_plot, loc1=2, loc2=3, fc = "none", ec='white')
    #plt.show()
    #save the plot as filename + "_some_string.png"
    plt.savefig("./slice_plots/" + str(file) + "_inset_plot.png")
    ds.index.clear_all_data()
    #file.close()
