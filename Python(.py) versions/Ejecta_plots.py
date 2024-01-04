#####################################################################################################################
# Python script to create slice plots panels. The script has multiple fuctionalities 
# to get the desired outputs simultentously as given below.
# 1: Script can create multi-panel slice plots 
# 2: Script can create custom fields (particle and cell)  and also read-in the data from TORCH nucleosynthetic yields.
# 3: Script also overplot scatter plot of particles on the slice plots. Particle size and color are based on 
#    The abundance of species.
# 4: Script also does the clump analysis using the Scipy.KDEtree 
# Written by Vrutant Mehta - 2023
#####################################################################################################################
# import required libraries. 
import yt
import glob
#import mpi4py
from unyt import unyt_array
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.colors as colors
from mpl_toolkits.axes_grid1 import AxesGrid
from matplotlib.ticker import ScalarFormatter
import numpy as np
import pandas as pd
#from scipy.interpolate import griddata
#import matplotlib.tri as tri
#import matplotlib.ticker as ticker
#import seaborn as sns
import cmasher as cmr
import scipy.spatial
#from scipy.stats import gaussian_kde
import math as math

from astropy.stats import jackknife_resampling
from astropy.stats import jackknife_stats

#custom colormaps from CMasher module
#mpl_cmap = cmr.get_sub_cmap('cmr.eclipse_r',0.00, 0.90)
rd_bu_cmap = cmr.get_sub_cmap('RdBu',0.00,1.00)
Bu_cmap = cmr.get_sub_cmap('Blues',0.00,1.00)
Rd_cmap = cmr.get_sub_cmap('Reds',0.00,1.00)

#custom class to write tick labels in Sci. notations. 
class ScalarFormatterForceFormat(ScalarFormatter):
    def _set_format(self):
        self.format = "%1.1f"

#define function for the Jackknife analysis
def JK_stats(array):
    arr = array.to_numpy()
    estimate, bias, error, conf_internval = jackknife_stats(arr, np.mean, 0.95)
    return estimate, bias, error, conf_internval

#define fuction for data reduction in a array (important when large arrays (>10e6 elements) are used.)
def factoring(arr):
    output = np.array(arr[::10000])
    return(output)

#default font and thier size unless changed. 
plt.rcParams['font.family'] = 'DejaVu Serif'
mpl.rcParams['font.sans-serif'] = ['Times']

#used in dertermining size of the domains
cent = 2.6e9

# provide files here
files = glob.glob ("def_std_hdf5_chk_000075")
files.sort()

#read in the .dat file that contains information of Particle tags, Iron mass fractions, Cromium mass fractions, and thier ratios
tag_file = pd.read_csv("./part_IGEs_Ti_SCD_PD_Z_0p0.dat", sep = "\s+",
                       usecols = [0, 1, 2, 3, 4, 5, 6, 7],
                       names = ["p_tag", "Fe", "Ni", "Mn", "Cr", "Ti", "Cr/Fe", "Ti/Fe"])

#assign seperate array for particle tags
tags = tag_file["p_tag"].to_numpy()

#function for filtering the unbound particles from original datasaet and create new particle fields
def unbound_particles(pfilter,data):
    p_tags = data[(pfilter.filtered_type, "particle_tag")]
    clumps = np.in1d(p_tags, tags)
    return yt.YTArray(clumps)

#the field is added here
yt.add_particle_filter("unbound_particles", function = unbound_particles, filtered_type = "all", requires = ["particle_tag"])



#for loop for files
for file in files:
    ds = yt.load(file)

# Below we define some required functions for calculations of unbound particles such as binding energy of particles,
# and also we add our data of iron mass fractions and cromium mass fractions of each unbound particles to create
# new fields taht can be read in yt and plot them. We also re-define the x-pos and y-pos fields to r-pos and z-pos
# since our geometry is cylindrical
###########===============================yt functions begins here=====================================##############

    def binding_energy(field, data):
        return(((0.5 * ((data[("flash", "velx")])**2 + (data[("flash", "vely")])**2)) +
                data[("flash", "eint")] + data[("flash", "gpot")]))

    ds.add_field(("flash", "bneg"),
                 function = binding_energy,
                 sampling_type = "cell",
                 units = "auto")

    ds.add_particle_filter("unbound_particles")

    def X_iron(field,data):
        A = data[("unbound_particles", "particle_tag")]
        A = A.tolist()
        filtered_X_fe = tag_file
        #filtered_X_fe["Fe"] = filtered_X_fe["Fe"].where(filtered_X_fe["Cr/Fe"] > 1.0e-2, other = 0)
        filtered_p_tag = filtered_X_fe.loc[filtered_X_fe['p_tag'].isin(A)]
        filtered_p_tag = filtered_p_tag.reset_index(drop = True)
        filtered_p_tag = filtered_p_tag.iloc[[i for i, x in sorted(enumerate(filtered_p_tag['p_tag']), key=lambda x: A.index(x[1]))], :]
        arr_fe = filtered_p_tag["Fe"].to_numpy()
        X_fe = unyt_array(arr_fe, "dimensionless")
        return (X_fe)

    ds.add_field(("unbound_particles", "X_fe"), function = X_iron, sampling_type = "particle", units = 'dimensionless')
    ds.add_deposited_particle_field(("unbound_particles", "X_fe"),'nearest', weight_field = "particle_ones",kernel_name = "cubic")

    def X_nickle(field,data):
        B = data[("unbound_particles", "particle_tag")]
        B = B.tolist()
        filtered_X_ni = tag_file
        #filtered_X_ni["Ni"] = filtered_X_ni["Ni"].where(filtered_X_ni["Cr/Fe"] > 1.0e-2, other = 0)
        filtered_p_tag = filtered_X_ni.loc[filtered_X_ni['p_tag'].isin(B)]
        filtered_p_tag = filtered_p_tag.reset_index(drop = True)
        filtered_p_tag = filtered_p_tag.iloc[[i for i, x in sorted(enumerate(filtered_p_tag['p_tag']), key=lambda x: B.index(x[1]))], :]
        arr_ni = filtered_p_tag["Ni"].to_numpy()
        X_ni = unyt_array(arr_ni, "dimensionless")
        return (X_ni)

    ds.add_field(("unbound_particles", "X_ni"), function = X_nickle, sampling_type = "particle", units = 'dimensionless')
    ds.add_deposited_particle_field(("unbound_particles", "X_ni"),'nearest', weight_field = "particle_ones",kernel_name = "cubic")

    def X_manganese(field,data):
        A = data[("unbound_particles", "particle_tag")]
        A = A.tolist()
        filtered_X_mn = tag_file
        #filtered_X_mn["Mn"] = filtered_X_mn["Mn"].where(filtered_X_mn["Cr/Fe"] > 1.0e-2, other = 0)
        filtered_p_tag = filtered_X_mn.loc[filtered_X_mn['p_tag'].isin(A)]
        filtered_p_tag = filtered_p_tag.reset_index(drop = True)
        filtered_p_tag = filtered_p_tag.iloc[[i for i, x in sorted(enumerate(filtered_p_tag['p_tag']), key=lambda x: A.index(x[1]))], :]
        arr_mn = filtered_p_tag["Mn"].to_numpy()
        X_mn = unyt_array(arr_mn, "dimensionless")
        return (X_mn)

    ds.add_field(("unbound_particles", "X_mn"), function = X_manganese, sampling_type = "particle", units = 'dimensionless')
    ds.add_deposited_particle_field(("unbound_particles", "X_mn"),'nearest', weight_field = "particle_ones",kernel_name = "cubic")


    def X_chromium(field, data):
        B = data[("unbound_particles", "particle_tag")]
        B = B.tolist()
        filtered_X_cr = tag_file
        #filtered_X_cr["Cr"] = filtered_X_cr["Cr"].where(filtered_X_cr["Cr/Fe"] > 1.0e-2, other = 0)
        filtered_p_tag = filtered_X_cr.loc[filtered_X_cr['p_tag'].isin(B)]
        filtered_p_tag = filtered_p_tag.reset_index(drop = True)
        filtered_p_tag = filtered_p_tag.iloc[[j for j, y in sorted(enumerate(filtered_p_tag['p_tag']), key=lambda y: B.index(y[1]))], :]
        arr_y = filtered_p_tag["Cr"].to_numpy()
        X_cr = unyt_array(arr_y, "dimensionless")
        return X_cr


    ds.add_field(("unbound_particles", "X_cr"), function = X_chromium, sampling_type = "particle", units = 'dimensionless')
    ds.add_deposited_particle_field(("unbound_particles","X_cr"), "nearest", weight_field = "particle_ones")

    def X_titanium(field, data):
        B = data[("unbound_particles", "particle_tag")]
        B = B.tolist()
        filtered_X_cr = tag_file
        #filtered_X_cr["Cr"] = filtered_X_cr["Cr"].where(filtered_X_cr["Cr/Fe"] > 1.0e-2, other = 0)
        filtered_p_tag = filtered_X_cr.loc[filtered_X_cr['p_tag'].isin(B)]
        filtered_p_tag = filtered_p_tag.reset_index(drop = True)
        filtered_p_tag = filtered_p_tag.iloc[[j for j, y in sorted(enumerate(filtered_p_tag['p_tag']), key=lambda y: B.index(y[1]))], :]
        arr_y = filtered_p_tag["Ti"].to_numpy()
        X_ti = unyt_array(arr_y, "dimensionless")
        return X_ti

    ds.add_field(("unbound_particles", "X_ti"), function = X_titanium, sampling_type = "particle", units = 'dimensionless')

    def X_Cr_to_Fe(field,data):
        C = data[("unbound_particles", "particle_tag")]
        C = C.tolist()
        filtered_X_cr_fe = tag_file
        #filtered_X_cr_fe["Cr/Fe"] = filtered_X_cr_fe["Cr/Fe"].where(filtered_X_cr_fe["Cr/Fe"] > 1.0e-2, other = 0)
        filtered_p_tag = filtered_X_cr_fe.loc[filtered_X_cr_fe['p_tag'].isin(C)]
        filtered_p_tag = filtered_p_tag.reset_index(drop = True)
        filtered_p_tag = filtered_p_tag.iloc[[k for k, z in sorted(enumerate(filtered_p_tag['p_tag']), key=lambda z: C.index(z[1]))], :]
        arr_z = filtered_p_tag["Cr/Fe"].to_numpy()
        X_cr_fe = unyt_array(arr_z, "dimensionless")
        return X_cr_fe

    ds.add_field(("unbound_particles", "X_Cr/Fe"), function = X_Cr_to_Fe, sampling_type = "particle", units = "dimensionless")
    ds.add_deposited_particle_field(("unbound_particles", "X_Cr/Fe"), "nearest", weight_field = "particle_ones")

    def X_Ti_to_Fe(field,data):
        C = data[("unbound_particles", "particle_tag")]
        C = C.tolist()
        filtered_X_cr_fe = tag_file
        #filtered_X_cr_fe["Cr/Fe"] = filtered_X_cr_fe["Cr/Fe"].where(filtered_X_cr_fe["Cr/Fe"] > 1.0e-2, other = 0)
        filtered_p_tag = filtered_X_cr_fe.loc[filtered_X_cr_fe['p_tag'].isin(C)]
        filtered_p_tag = filtered_p_tag.reset_index(drop = True)
        filtered_p_tag = filtered_p_tag.iloc[[k for k, z in sorted(enumerate(filtered_p_tag['p_tag']), key=lambda z: C.index(z[1]))], :]
        arr_z = filtered_p_tag["Ti/Fe"].to_numpy()
        X_ti_fe = unyt_array(arr_z, "dimensionless")
        return X_ti_fe

    ds.add_field(("unbound_particles", "X_Ti/Fe"), function = X_Ti_to_Fe, sampling_type = "particle", units = "dimensionless")

    def part_pos_r(field, data):
        return (data["unbound_particles", "particle_posx"])
    ds.add_field(("unbound_particles", "particle_position_r"), function = part_pos_r, sampling_type = "particle", units = "auto")

    def part_pos_z(field, data):
        return (data["unbound_particles", "particle_posy"])
    ds.add_field(("unbound_particles", "particle_position_z"), function = part_pos_z, sampling_type = "particle", units = "auto", force_override = True)

    
###########===============================functions ends here=====================================##############
    
    
    fig = plt.figure() # figure for multipanel
    # Axes grid object from matplotlib is used to create empty panel to store the images
    grid = AxesGrid(fig,rect = (0.075, 0.075, 3, 1),
                    nrows_ncols = (1, 3),
                    axes_pad= 2.0,
                    label_mode="2",
                    share_all = True,
                    cbar_location = "right",
                    cbar_mode = "single",
                    cbar_size = "6%",
                    cbar_pad = "10%")
    
    #basic yt functions for loading the data and some data_objects or data_containers to plot fields from the containers.
    ad = ds.all_data()
    box = ds.region(center = [cent,0,0], left_edge = [0,-2*cent,0], right_edge = [2*cent,2*cent,2*cent])
    bneg_new = ad.exclude_inside(("flash", "bneg"), -1.0e10, 1.0e10)
    bn_eg_without_fluff = bneg_new.exclude_below(("gas", "density"), 1.0e-2)

    weight = factoring(np.array(bn_eg_without_fluff[("flash","bneg")]))
    cont_r = factoring(np.array(bn_eg_without_fluff[("flash","r")]*1.0e-5))
    cont_z = factoring(np.array(bn_eg_without_fluff[("flash","z")]*1.0e-5))
    
    #print(len(cont_r), len(cont_z), len(weight))
    #negative_binding_energy = ad.cut_region(['obj["flash", "bneg"] < -1.0e15'])

    #sliceplot begins here with some plot args such are colormap, colorbar limits, fields to plot, etc.
    slc1 = yt.SlicePlot(ds, 'theta', ('gas', 'density'), center=[cent,0,0], data_source = box)
    slc1.set_zlim(('gas', 'density'), 1.0e-2, 1.0e8)
    slc1.set_cmap (field = ('gas', 'density'), cmap = "plasma")
    #slc1.annotate_particles(width = (4*cent, "cm"), p_size=5.0, col='k', marker='o', ptype='unbound_particles',alpha=1.0)
    #slc1.annotate_title("Density Plot with 'Iron' clumps in contours")
    slc1.set_font_size(26)
    #slc1.annotate_contour(field = ('flash', 'bneg'), levels = 1, clim = [1.0e-10,1.0e-10], take_log = False, plot_args = {"colors":"black", "linewidths": 2.0})
    slc1.annotate_timestamp (corner='upper_right', draw_inset_box=True, redshift = False)
    slc1.set_width((2*cent, 4*cent))
    slc1.hide_axes()
    #slc1.set_figure_size(20)
    #slc1.set_buff_size(1600)
    #slc1.show()
    
    #same as above
    slc2 = yt.SlicePlot(ds, 'theta', ("gas", "density"), center=[cent,0,0], data_source = box)
    slc2.set_zlim(('gas', 'density'), 1.0e-2, 1.0e8)
    slc2.set_cmap (field = ("gas", "density"), cmap = "plasma")
    #slc2.annotate_particles(width = (4*cent, "cm"), p_size=5.0, col='k', marker='o', ptype='unbound_particles',alpha=1.0)
    #slc2.annotate_title("Density Plot with 'Chromium' Clumps")
    slc2.set_font_size(26)
    #slc2.annotate_contour(field = ('flash', 'bneg'), levels = 1, clim = [1.0e-10,1.0e-10], take_log = False, plot_args = {"colors":"black", "linewidths": 2.0})
    slc2.annotate_timestamp (corner='upper_right', draw_inset_box=True, redshift = False)
    slc2.set_width((2*cent, 4*cent))
    slc2.hide_axes()
    #slc2.set_figure_size(20)
    #slc2.set_buff_size(1600)
    #slc1.show()
    
    #same as above
    slc3 = yt.SlicePlot(ds, 'theta', ("gas", "density"), center=[cent,0,0], data_source = box)
    slc3.set_zlim(('gas', 'density'), 1.0e-2, 1.0e8)
    slc3.set_cmap (field = ("gas", "density"), cmap = "plasma")
    #slc3.annotate_particles(width = (4*cent, "cm"), p_size=5.0, col='k', marker='o', ptype='unbound_particles',alpha=1.0)
    #slc3.annotate_title("Density Plot with High 'Cr/Fe' mass fractions")
    slc3.set_font_size(26)
    #slc3.annotate_contour(field = ('flash', 'bneg'), levels = 1, clim = [1.0e-10,1.0e-10],take_log = False, plot_args = {"colors":"black", "linewidths": 2.0})
    slc3.annotate_timestamp (corner='upper_right', draw_inset_box=True, redshift = False)
    slc3.set_width((2*cent, 4*cent))
    slc3.hide_axes()
    #slc3.set_figure_size(20)
    #slc3.set_buff_size(1600)
    #slc3.show()
    

    #X_Fe = np.array(ad[('unbound_particles','X_fe')])
    #X_Cr = np.array(ad[('unbound_particles','X_cr')])
    #X_Cr_to_Fe = X_Cr/X_Fe
    
    #X_Cr_to_Fe[np.isnan(X_Cr_to_Fe)] = 0

    df = pd.DataFrame({'X_Fe': np.array(ad[('unbound_particles','X_fe')]),
                       'X_Ni': np.array(ad[('unbound_particles','X_ni')]),
                       'X_Mn': np.array(ad[('unbound_particles','X_mn')]),
                       'X_Cr': np.array(ad[('unbound_particles','X_cr')]),
                       'X_Ti': np.array(ad[('unbound_particles','X_ti')]),
                       'Cr_to_Fe_ratio' : np.array(ad[('unbound_particles', 'X_Cr/Fe')]),
                       'Ti_to_Fe_ratio' : np.array(ad[('unbound_particles', 'X_Ti/Fe')]),
                       'R_pos': np.array((ad["unbound_particles", "particle_posx"])*1.0e-5),
                       'Z_pos': np.array((ad["unbound_particles", "particle_posy"])*1.0e-5)})

    #sort them in descending order in ratios of Cr mass fractions to Fe mass fractions (X_Cr/X_Fe) and reset index.
    df.sort_values(by = ["Cr_to_Fe_ratio"], ascending = False, inplace = True)
    df = df.reset_index(drop = True)
    #df.to_csv("particles_data_Z0.dat", sep = " ", header = False, index = False, float_format = '%g')

#====================---------         clump analysis begins here         ---------========================#
    
    #sort dataframe in descending order of Cr/Fe ratio such that first element in dataframe has highest Cr/Fe ratio.
    df.sort_values(by = ["Cr_to_Fe_ratio"], ascending = False, inplace = True)
    #reset the index of the dataframe
    df = df.reset_index(drop = True)

    #Jacknife analysis of all the abundances that we report
    G_Fe_est, G_Fe_bias, G_Fe_err, G_Fe_interval = JK_stats(df["X_Fe"])
    G_Ni_est, G_Ni_bias, G_Ni_err, G_Ni_interval = JK_stats(df["X_Ni"])
    G_Mn_est, G_Mn_bias, G_Mn_err, G_Mn_interval = JK_stats(df["X_Mn"])
    G_Cr_est, G_Cr_bias, G_Cr_err, G_Cr_interval = JK_stats(df["X_Cr"])
    G_Ti_est, G_Ti_bias, G_Ti_err, G_Ti_interval = JK_stats(df["X_Ti"])

    #print out the ratios of our Jack-knife analysis. such as ratios of mean, error and confidence intervals.
    #print("Est. :" , Cr_est/Fe_est)
    #print("Error :", Cr_err/Fe_err)
    #print("Conf. interval :", Cr_interval/Fe_interval)

    print("Global Ni/Fe Est. :" , G_Ni_est/G_Fe_est)
    print("Global Mn/Fe Est. :" , G_Mn_est/G_Fe_est)
    print("Global Cr/Fe Est. :" , G_Cr_est/G_Fe_est)
    print("Global Ti/Fe Est. :" , G_Ti_est/G_Fe_est)

    print("Global Ni/Fe Std. Error :" , G_Ni_err/G_Fe_err)
    print("Global Mn/Fe Std. Error :" , G_Mn_err/G_Fe_err)
    print("Global Cr/Fe Std. Error :" , G_Cr_err/G_Fe_err)
    print("Global Ti/Fe Std. Error :" , G_Ti_err/G_Fe_err)

    print("Global Ni/Fe Conf. interval :" , G_Ni_interval/G_Fe_interval)
    print("Global Mn/Fe Conf. interval :" , G_Mn_interval/G_Fe_interval)
    print("Global Cr/Fe Conf. interval :" , G_Cr_interval/G_Fe_interval)
    print("Global Ti/Fe Conf. interval :" , G_Ti_interval/G_Fe_interval)

    print("Global Ni/Fe upper err. bar :" , (G_Ni_interval[1]/G_Fe_interval[1]) - (G_Ni_est/G_Fe_est))
    print("Global Mn/Fe upper err. bar :" , (G_Mn_interval[1]/G_Fe_interval[1]) - (G_Mn_est/G_Fe_est))
    print("Global Cr/Fe upper err. bar :" , (G_Cr_interval[1]/G_Fe_interval[1]) - (G_Cr_est/G_Fe_est))
    print("Global Ti/Fe upper err. bar :" , (G_Ti_interval[1]/G_Fe_interval[1]) - (G_Ti_est/G_Fe_est))

    print("Global Ni/Fe lower err. bar :" , (G_Ni_interval[0]/G_Fe_interval[0]) - (G_Ni_est/G_Fe_est))
    print("Global Mn/Fe lower err. bar :" , (G_Mn_interval[0]/G_Fe_interval[0]) - (G_Mn_est/G_Fe_est))
    print("Global Cr/Fe lower err. bar :" , (G_Cr_interval[0]/G_Fe_interval[0]) - (G_Cr_est/G_Fe_est))
    print("Global Ti/Fe lower err. bar :" , (G_Ti_interval[0]/G_Fe_interval[0]) - (G_Ti_est/G_Fe_est))

    print("Global Ni/Fe err. in % :", ((G_Ni_interval[1]/G_Fe_interval[1]) - (G_Ni_est/G_Fe_est))/(G_Ni_est/G_Fe_est), 
                                      ((G_Ni_interval[0]/G_Fe_interval[0]) - (G_Ni_est/G_Fe_est))/(G_Ni_est/G_Fe_est))
    print("Global Mn/Fe err. in % :", ((G_Mn_interval[1]/G_Fe_interval[1]) - (G_Mn_est/G_Fe_est))/(G_Mn_est/G_Fe_est), 
                                      ((G_Mn_interval[0]/G_Fe_interval[0]) - (G_Mn_est/G_Fe_est))/(G_Mn_est/G_Fe_est)) 
    print("Global Cr/Fe err. in % :", ((G_Cr_interval[1]/G_Fe_interval[1]) - (G_Cr_est/G_Fe_est))/(G_Cr_est/G_Fe_est), 
                                      ((G_Cr_interval[0]/G_Fe_interval[0]) - (G_Cr_est/G_Fe_est))/(G_Cr_est/G_Fe_est))
    print("Global Ti/Fe err. in % :", ((G_Ti_interval[1]/G_Fe_interval[1]) - (G_Ti_est/G_Fe_est))/(G_Ti_est/G_Fe_est), 
                                      ((G_Ti_interval[0]/G_Fe_interval[0]) - (G_Ti_est/G_Fe_est))/(G_Ti_est/G_Fe_est)) 

    #assign seperate dateframe for z (Cr/Fe ratios), x (R-position), y (Z-position)
    #please note that here x is cylindrical R-coordinates and y is cylindrical Z-coordinates of particles.
    Cr_to_Fe = df["Cr_to_Fe_ratio"]
    x = df['R_pos']
    y = df['Z_pos']

    #get n = length of the dataframe
    n = len(df)

    # empty lists to store the analyzed data.
    clump_array = []
    Ni_to_Fe_JKS = []
    Mn_to_Fe_JKS = []
    Cr_to_Fe_JKS = []
    Ti_to_Fe_JKS = []

    #first for loop to analyze all the particles
    for j in range(0,n):
        print(j)
        p = Cr_to_Fe[j] #p is variable that is Cr/Fe ratio of the point(particle) that we are analyzing in the loop
                        #p begins from 0 that is first entry in Cr/Fe dataframe and it loop over n particles


        x0 = x[j]    # x0 is similar as above, the R-coord. of our particle that we are analyzing
        y0 = y[j]    # y0 is similar as above, the Z-coord. of our particle that we are analyzing

        points = list(zip(x,y)) # we create the list that has of pairs of R and Z coordinates of all particles

        ckdtree = scipy.spatial.cKDTree(points) # Scipy function to create KD Tree to lookup the nearest neighbour.
                                                # for more info please look up at
                                                # https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.cKDTree.html

        # make sure these lists are empty before they are used again in the loop below
        clump_points = [] # used to store the data (distances and index) of the nearest neighbor points.
        clump_data = []   # used for creating a seperate dataframe for the particles identified in the clump.


        #next for loop that looks up into nearest neighbour of our particle 'p' from ckdtree created
        for increment in range(1,n+1):
            clump_points = ckdtree.query([x0,y0],increment) # ckdtree.query takes in two arguments 1st is x and y position of our
                                                            # reference point or the point that we are analyzing (p) and 2nd argument is
                                                            # number of nearest neighbours to consider
                                                            # i.e. 1 = point itself, 2 = point itself and nearest point to it and so on.
                                                            # this returns list of two arrays.
                                                            # 1st array is the distances of nearest neighbor points from our referance point.
                                                            # 2nd array is the index of those nearest neighbor points.
                                                            # to read more - https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.cKDTree.query.html#scipy.spatial.cKDTree.query

            clump_data = df.iloc[clump_points[1]]  # here we use 2nd array of clump_points which contains the index of particles in our clump.
                                                   # we locate these index on our orignal 'df' dataframe and filter those particles data as
                                                   # new dataframe called clump_data.
                                                   # clump_data has same columns as df dataframe, but for our filtered particles.

            if (clump_data["X_Fe"].sum()) >= 0.07 * (df["X_Fe"].sum()):  # the condition for termination.
                                                                             # if sum of X_fe for all the particles in the clump
                                                                             # that we are indentifying,
                                                                             # is > or = 0.07 of sum of X_fe in ejecta. loop terminates.

                    print(clump_data['X_Fe'].sum()/df['X_Fe'].sum())         # print to verify.

                    break # break the loop


        #At this point we have a data from our clump and we can begin the analysis using clump_data

        #Jacknife analysis of all the abundances that we report
        Fe_est, Fe_bias, Fe_err, Fe_interval = JK_stats(clump_data["X_Fe"])
        Ni_est, Ni_bias, Ni_err, Ni_interval = JK_stats(clump_data["X_Ni"])
        Mn_est, Mn_bias, Mn_err, Mn_interval = JK_stats(clump_data["X_Mn"])
        Cr_est, Cr_bias, Cr_err, Cr_interval = JK_stats(clump_data["X_Cr"])
        Ti_est, Ti_bias, Ti_err, Ti_interval = JK_stats(clump_data["X_Ti"])

        #print out the ratios of our Jack-knife analysis. such as ratios of mean, error and confidence intervals.
        #print("Est. :" , Cr_est/Fe_est)
        #print("Error :", Cr_err/Fe_err)
        #print("Conf. interval :", Cr_interval/Fe_interval)

        print("Ni/Fe Est. :" , Ni_est/Fe_est)
        print("Mn/Fe Est. :" , Mn_est/Fe_est)
        print("Cr/Fe Est. :" , Cr_est/Fe_est)
        print("Ti/Fe Est. :" , Ti_est/Fe_est)

        print("Ni/Fe Std. Error :" , Ni_err/Fe_err)
        print("Mn/Fe Std. Error :" , Mn_err/Fe_err)
        print("Cr/Fe Std. Error :" , Cr_err/Fe_err)
        print("Ti/Fe Std. Error :" , Ti_err/Fe_err)

        print("Ni/Fe Conf. interval :" , Ni_interval/Fe_interval)
        print("Mn/Fe Conf. interval :" , Mn_interval/Fe_interval)
        print("Cr/Fe Conf. interval :" , Cr_interval/Fe_interval)
        print("Ti/Fe Conf. interval :" , Ti_interval/Fe_interval)

        print("Clump Ni/Fe upper err. bar :" , (Ni_interval[1]/Fe_interval[1]) - (Ni_est/Fe_est))
        print("Clump Mn/Fe upper err. bar :" , (Mn_interval[1]/Fe_interval[1]) - (Mn_est/Fe_est))
        print("Clump Cr/Fe upper err. bar :" , (Cr_interval[1]/Fe_interval[1]) - (Cr_est/Fe_est))
        print("Clump Ti/Fe upper err. bar :" , (Ti_interval[1]/Fe_interval[1]) - (Ti_est/Fe_est))

        print("Clump Ni/Fe lower err. bar :" , (Ni_interval[0]/Fe_interval[0]) - (Ni_est/Fe_est))
        print("Clump Mn/Fe lower err. bar :" , (Mn_interval[0]/Fe_interval[0]) - (Mn_est/Fe_est))
        print("Clump Cr/Fe lower err. bar :" , (Cr_interval[0]/Fe_interval[0]) - (Cr_est/Fe_est))
        print("Clump Ti/Fe lower err. bar :" , (Ti_interval[0]/Fe_interval[0]) - (Ti_est/Fe_est))

        print("Global Ni/Fe err. in % :", ((Ni_interval[1]/Fe_interval[1]) - (Ni_est/Fe_est))/(Ni_est/Fe_est),
                                          ((Ni_interval[0]/Fe_interval[0]) - (Ni_est/Fe_est))/(Ni_est/Fe_est))
        print("Global Mn/Fe err. in % :", ((Mn_interval[1]/Fe_interval[1]) - (Mn_est/Fe_est))/(Mn_est/Fe_est), 
                                          ((Mn_interval[0]/Fe_interval[0]) - (Mn_est/Fe_est))/(Mn_est/Fe_est))
        print("Global Cr/Fe err. in % :", ((Cr_interval[1]/Fe_interval[1]) - (Cr_est/Fe_est))/(Cr_est/Fe_est),  
                                          ((Cr_interval[0]/Fe_interval[0]) - (Cr_est/Fe_est))/(Cr_est/Fe_est))
        print("Global Ti/Fe err. in % :", ((Ti_interval[1]/Fe_interval[1]) - (Ti_est/Fe_est))/(Ti_est/Fe_est),  
                                          ((Ti_interval[0]/Fe_interval[0]) - (Ti_est/Fe_est))/(Ti_est/Fe_est))

        fst_pt_r = clump_data["R_pos"].iloc[0]        # R-coord. of our reference point
        fst_pt_z = clump_data["Z_pos"].iloc[0]        # Z-coord. of our referance point

        lst_pt_r = clump_data["R_pos"].iloc[-1]       # R-ccord. of last nearest neighbor point that was added to the clump
        lst_pt_z = clump_data["Z_pos"].iloc[-1]       # Z-ccord. of last nearest neighbor point that was added to the clump

        r_dist = abs(lst_pt_r - fst_pt_r)                   # distance in R-direction between two points
        z_dist = abs(lst_pt_z - fst_pt_z)                   # distance in Z-direction between two points
        radius_dist = math.sqrt((r_dist**2) + (z_dist**2))  # actual distance between two points
        print("%.5e" %fst_pt_r,"km", "%.5e" %fst_pt_z, "km", "%.5e" %radius_dist, "km") # print center-point's coordinates and the radius.

        total_Fe = np.sum(clump_data["X_Fe"])
        total_Cr = np.sum(clump_data["X_Cr"])
        mean_Fe = np.mean(clump_data["X_Fe"])
        mean_Cr = np.mean(clump_data["X_Cr"])
        clump_Cr_to_Fe = mean_Cr/mean_Fe

        print("total Fe in the clump: ", "%.5f" %total_Fe) #print sum of mass fractions of Fe in the clump particles
        print("total Cr in the clump: ", "%.5f" %total_Cr) #print sum of mass fractions of Cr in the clump particles
        print("mean Fe in the clump: ", "%.5f" %mean_Fe)  #print mean_Fe
        print("mean Cr in the clump: ","%.5f" %mean_Cr)   #print mean_Cr
        print("Cr/Fe in the clump : ", "%.5f" %clump_Cr_to_Fe) #print ratio of Cr/Fe
        print() #empty print to seperate consicutive loop's prints

        #store the important data into array and use this at very end to plot
        #clump_array.append([total_Fe,
        #                    total_Cr,
        #                    mean_Fe,
        #                    mean_Cr,
        #                    clump_Cr_to_Fe,
        #                    fst_pt_r,
        #                    fst_pt_z,
        #                    radius_dist])
    #create dataframe from the list clump_array since it is easier to work with.
    #clumps_df = pd.DataFrame(clump_array,
    #                         columns = ['total_Fe',
    #                                    'total_Cr',
    #                                    'mean_Fe',
    #                                    'mean_Cr',
    #                                    'clump_Cr_to_Fe',
    #                                    'fst_pt_r',
    #                                    'fst_pt_z',
    #                                    'radius_dist'],
    #                         dtype = float)

    #sort it in descending order of clump_Cr_to_Fe column
    #clumps_df.sort_values(by = ["clump_Cr_to_Fe"], ascending = False, inplace = True)

    #print the dataframe
    #print(clumps_df.to_string())

    #clumps_df = clumps_df.reset_index(drop = True)

        clump_array.append([len(clump_data),
                            np.mean(clump_data['X_Fe']),
                            np.mean(clump_data['X_Ni']),
                            np.mean(clump_data['X_Mn']),
                            np.mean(clump_data['X_Cr']),
                            np.mean(clump_data['X_Ti']),
                            (np.mean(clump_data['X_Ni']))/(np.mean(clump_data['X_Fe'])),
                            (np.mean(clump_data['X_Mn']))/(np.mean(clump_data['X_Fe'])),
                            (np.mean(clump_data['X_Cr']))/(np.mean(clump_data['X_Fe'])),
                            (np.mean(clump_data['X_Ti']))/(np.mean(clump_data['X_Fe'])),
                            fst_pt_r,
                            fst_pt_z,
                            radius_dist])
        Ni_to_Fe_JKS.append([(Ni_est/Fe_est),
                            (Ni_err/Fe_err),
                            (Ni_interval/Fe_interval),
                            ((Ni_interval[1]/Fe_interval[1]) - Ni_est/Fe_est),
                            ((Ni_interval[0]/Fe_interval[0]) - Ni_est/Fe_est)])

        Mn_to_Fe_JKS.append([(Mn_est/Fe_est),
                            (Mn_err/Fe_err),
                            (Mn_interval/Fe_interval),
                            ((Mn_interval[1]/Fe_interval[1]) - Mn_est/Fe_est),
                            ((Mn_interval[0]/Fe_interval[0]) - Mn_est/Fe_est)])

        Cr_to_Fe_JKS.append([(Cr_est/Fe_est),
                            (Cr_err/Fe_err),
                            (Cr_interval/Fe_interval),
                            ((Cr_interval[1]/Fe_interval[1]) - Cr_est/Fe_est),
                            ((Cr_interval[0]/Fe_interval[0]) - Cr_est/Fe_est)])

        Ti_to_Fe_JKS.append([(Ti_est/Fe_est),
                            (Ti_err/Fe_err),
                            (Ti_interval/Fe_interval),
                            ((Ti_interval[1]/Fe_interval[1]) - Ti_est/Fe_est),
                            ((Ti_interval[0]/Fe_interval[0]) - Ti_est/Fe_est)])


    clumps_df = pd.DataFrame(clump_array,
                             columns = ['parts', 'mean_fe', 'mean_ni', 'mean_mn', 'mean_cr', 'mean_ti', 'ni_to_fe', 'mn_to_fe', 'cr_to_fe', 'ti_to_fe', 'fst_pt_r', 'fst_pt_z', 'radius_dist'],
                             dtype = float)

    clumps_df.sort_values(by = 'cr_to_fe', ascending = False, inplace = True)
    clumps_df.to_csv('clumps_details_Z0.dat', sep = ' ',header = True, index = True, float_format = '%g')
    print(clumps_df.to_string())

    Ni_df = pd.DataFrame(Ni_to_Fe_JKS,columns = ['mean', 'error', 'interval', 'upper_bar', 'lower_bar'], dtype = float)
    Ni_df = Ni_df.reindex(clumps_df.index)
    Ni_df.to_csv('clumps_details_Ni_Z0.dat', sep = ' ',header = True, index = True, float_format = '%g')

    Mn_df = pd.DataFrame(Mn_to_Fe_JKS,columns = ['mean', 'error', 'interval', 'upper_bar', 'lower_bar'], dtype = float)
    Mn_df = Mn_df.reindex(clumps_df.index)
    Mn_df.to_csv('clumps_details_Mn_Z0.dat', sep = ' ',header = True, index = True, float_format = '%g')

    Cr_df = pd.DataFrame(Cr_to_Fe_JKS,columns = ['mean', 'error', 'interval', 'upper_bar', 'lower_bar'], dtype = float)
    Cr_df = Cr_df.reindex(clumps_df.index)
    Cr_df.to_csv('clumps_details_Cr_Z0.dat', sep = ' ',header = True, index = True, float_format = '%g')

    Ti_df = pd.DataFrame(Ti_to_Fe_JKS,columns = ['mean', 'error', 'interval', 'upper_bar', 'lower_bar'], dtype = float)
    Ti_df = Ti_df.reindex(clumps_df.index)
    Ti_df.to_csv('clumps_details_Ti_Z0.dat', sep = ' ',header = True, index = True, float_format = '%g')


#====================------------        clump analysis ends here       -------------========================#
        #scatter_plot.add_patch(plt.Circle((clump_data['x_pos'].iloc[0], clump_data['y_pos'].iloc[0]), clump_data[str(v)].iloc[-1], fill = False))



    slc_plots = [(slc1, 0, df["X_Fe"], r'$\mathtt{X_{Fe}}$'), 
                 (slc2, 1, df["X_Cr"], r'$\mathtt{X_{Cr}}$'), 
                 (slc3, 2, df["Cr_to_Fe_ratio"], r'$\mathtt{\frac{X_{Cr}}{X_{Fe}}}$')]
    
    for slc, i, color, label in slc_plots:
        
        plot = slc.plots[("gas","density")]
        plot.figure = fig
        plot.axes = grid[i].axes
        plot.cax = grid.cbar_axes[i]
        slc.render()
        axes_locations = grid[i].get_axes_locator()
        
        x0, y0, w0, h0 = grid[i].get_position().bounds
        scatter_plot = fig.add_axes((x0, y0, w0, h0),facecolor = "None")
        X_pos = (ad["unbound_particles", "particle_posx"])*1.0e-5
        Y_pos = (ad["unbound_particles", "particle_posy"])*1.0e-5
        
        #seaborn_data = pd.DataFrame({'X Position': X_pos, 'Y Position': Y_pos, 'Z Value': np.log10(color/color.min())})

        if i == 2:
            #sns.kdeplot(data=seaborn_data, x='X Position', y='Y Position', weights='Z Value', levels=4, cmap="RdBu", fill=False)
            sp = scatter_plot.scatter(X_pos, Y_pos, c = color , marker = 'o',cmap = rd_bu_cmap, alpha = 1.0, s = (60 * ((color - color.min())/color.max() - color.min())) + 0.5 ,
                                      norm = colors.LogNorm(vmax = 1.0e5, vmin = 1.0e-2))
        if i == 1:
            #sns.kdeplot(data=seaborn_data, x='X Position', y='Y Position', weights='Z Value', levels=4, cmap="Blues", fill=False)
            sp = scatter_plot.scatter(X_pos, Y_pos,c = color, marker = 'o', cmap = Bu_cmap, alpha = 1.0, s = (60 * ((color - color.min())/color.max() - color.min())) + 0.5 , 
                                      norm = colors.LogNorm(vmax = 1.0e0, vmin = 1.0e-3))
        if i == 0:
            #sns.kdeplot(data=seaborn_data, x='X Position', y='Y Position', weights='Z Value', levels=4, cmap="Reds", fill=False)
            sp = scatter_plot.scatter(X_pos, Y_pos,c = color, marker = 'o', cmap = Rd_cmap, alpha = 1.0, s = (60 * ((color - color.min())/color.max() - color.min())) + 0.5 , 
                                      norm = colors.LogNorm(vmax = 1.0e0, vmin = 1.0e-3))

        xx0 ,xx1 = grid[i].axes.get_xlim()
        yy0 ,yy1 = grid[i].axes.get_ylim()
        scatter_plot.set_xlim(xx0,xx1)
        scatter_plot.set_ylim(yy0,yy1)
        scatter_plot.set_axes_locator(axes_locations)
        yfmt = ScalarFormatterForceFormat()
        yfmt.set_powerlimits((0,0))
        scatter_plot.yaxis.set_major_formatter(yfmt)
        scatter_plot.xaxis.set_major_formatter(yfmt)
        scatter_plot.ticklabel_format(axis = 'both', style = 'sci', useMathText= True, scilimits = [0,0])
        scatter_plot.tick_params(axis='both',labelsize = 20)
        scatter_plot.xaxis.get_offset_text().set_fontsize(20)
        scatter_plot.yaxis.get_offset_text().set_fontsize(20)
        scatter_plot.set_xlabel('r (km)', fontsize = 20)
        scatter_plot.set_ylabel('z (km)', fontsize = 20)
        cont = scatter_plot.tricontour(cont_r,cont_z, weight, levels = [0.0], linestyles = ["dashed"], colors= ["black"], linewidths = [2.0])
        scatter_plot.add_patch(plt.Circle((clumps_df['fst_pt_r'].iloc[0], clumps_df['fst_pt_z'].iloc[0]), 
                                           clumps_df['radius_dist'].iloc[0], fill = False, linewidth = 2.0,color ='white'))
        if i == 2:
            legend = scatter_plot.legend(cont.legend_elements()[0],['Bound Remnant', 'Marginally Bound Ejecta'],loc = 'lower right',  fontsize = 12)
            scatter_plot.add_artist(legend)
        #scatter_plot.add_patch(plt.Circle((clump_data['x_pos'].iloc[0], clump_data['y_pos'].iloc[0]), clump_data[str(v)].iloc[-1], fill = False, linewidth = 2.0, color = 'white'))
        #scatter_plot.axis("off")
        
        bar = grid[i].inset_axes([0.0,-0.15,1,0.02])
        c_bar = fig.colorbar(sp, cax = bar, orientation = 'horizontal', extend = 'min')
        c_bar.ax.tick_params(labelsize = 20)
        c_bar.set_label(label = label, size = 26) 
        
        slc.render()
    plt.show()
    fig.savefig("ejecta_plot_SCD_PD_Z_0p0.png",bbox_inches="tight")
    
