import yt
import glob
from unyt import unyt_array
#import cmasher as cmr
#from palettable.cmocean.sequential import Thermal_20

import matplotlib.ticker as ticker
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.colors as colors
from mpl_toolkits.axes_grid1 import AxesGrid
from matplotlib.ticker import ScalarFormatter
import seaborn as sns

import numpy as np
import pandas as pd
import scipy.spatial
import math as math

from astropy.stats import jackknife_resampling
from astropy.stats import jackknife_stats

#cmr_cmap = cmr.arctic
#mpl_cmap = plt.get_cmap('cmr.arctic')

#mpl_cmap = sns.color_palette("mako", as_cmap=True)

#Define a Class for fomatter in the axis of figure so that all of the dimensions are written in uniform manner.

class ScalarFormatterForceFormat(ScalarFormatter):
    def _set_format(self):
        self.format = "%1.1f"
#define function for the Jackknife analysis
def JK_stats(array):
    arr = array.to_numpy()
    estimate, bias, error, conf_internval = jackknife_stats(arr, np.mean, 0.95)
    return estimate, bias, error, conf_internval


#set same type of fonts for figure
plt.rcParams['font.family'] = 'DejaVu Serif'
mpl.rcParams['font.sans-serif'] = ['Times']

#contants for calculations such as center of the domain, mass of star and number of particles 
cent = 2.6e9
mass = 2.78258e33
particles = 9990
mass_per_particle = mass/particles
#n = 5
#ejecta_particles = len(data)
#ejecta_mass = ejecta_particles * mass_per_particle


## function for Jackknife resampling (Not required since Astropy has the Jackknife library)
#def jackknife_resampling(Arr):
#    Arr = Arr.to_numpy()
#    nw_Arr = []
#    for i in range(len(Arr)):
#        array = np.delete(Arr, [i], axis = 0)
#        nw_Arr.append(np.sum(array,axis=0))
#    return nw_Arr
        
#read in files
files = glob.glob ("def_high_dens_hdf5_chk_000084")
files.sort()

#read in the .dat file that contains information of Particle tags, Iron mass fractions, Cromium mass fractions, and thier ratios
tag_file = pd.read_csv("./part_IGEs_Ti_HCD_PD_Z_0p0.dat", sep = "\s+", 
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
        return(data["gas", "density"] * 
               ((0.5 * ((data[("flash", "velx")])**2 + (data[("flash", "vely")])**2)) + 
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
    
###########=============================== yt functions ends here =====================================##############
    #plot begins here
    fig = plt.figure()
    #axes grid is required if we are plotting more than 1 plots simulteneously, Please read yt docs. for more details
    grid = AxesGrid(fig,rect = (0.075, 0.075, 1, 1),
                    nrows_ncols = (1, 1),
                    axes_pad= 2.0,
                    label_mode="2",
                    share_all = True,
                    cbar_location = "right",
                    cbar_mode = "single",
                    cbar_size = "6%",
                    cbar_pad = "10%")
    
    #load all data to ad dataset.
    ad = ds.all_data()
    slc_arr = [] #using this list to analyse multiple clumps.
    
    #for loop for creating slice plot figure
    for f in range(0,1):
        slc = yt.SlicePlot(ds, 'theta', ('gas', 'density'), center=[cent,0,0])
        slc.set_zlim(('gas', 'density'), 1.0e-2, 1.0e8)
        slc.set_cmap (field = ('gas', 'density'), cmap = 'plasma')
        #slc.annotate_particles(width = (4*cent, "cm"), p_size=5.0, col='k', marker='o', ptype='unbound_particles',alpha=1.0)
        #slc.annotate_title("Density Plot with 'Iron' clumps in contours")
        slc.set_font_size(26)
        #slc.annotate_contour(field = ('deposit', 'unbound_particles_nn_X_fe'),levels = 3,take_log = False ,plot_args = {"colors":"red", "linewidths": 0.5})
        slc.annotate_timestamp (corner='upper_right', draw_inset_box=True, redshift = False)
        slc.set_width((2*cent, 4*cent))
        slc.hide_axes()
        slc_arr.append([slc,f]) #we append this plot to list above with a index of the iteration.
    
    
    #create new dataframe so that everything in same order as in the yt dataset including x-positon and y-position of unbound particles.
    df = pd.DataFrame({'X_Fe': np.array(ad[('unbound_particles','X_fe')]),
                       'X_Ni': np.array(ad[('unbound_particles','X_ni')]),
                       'X_Mn': np.array(ad[('unbound_particles','X_mn')]), 
                       'X_Cr': np.array(ad[('unbound_particles','X_cr')]),
                       'X_Ti': np.array(ad[('unbound_particles','X_ti')]),
                       'Cr_to_Fe_ratio' : np.array(ad[('unbound_particles', 'X_Cr/Fe')]),
                       'Ti_to_Fe_ratio' : np.array(ad[('unbound_particles', 'X_Ti/Fe')]),
                       'x_pos': np.array((ad["unbound_particles", "particle_posx"])*1.0e-5),
                       'y_pos': np.array((ad["unbound_particles", "particle_posy"])*1.0e-5)})
    
    #sort them in descending order in ratios of Cr mass fractions to Fe mass fractions (X_Cr/X_Fe) and reset index.
    df.sort_values(by = ["Cr_to_Fe_ratio"], ascending = False, inplace = True)
    df = df.reset_index(drop = True)
    df.to_csv("particles_data_Z0.dat", sep = " ", header = False, index = False, float_format = '%g')

    #assign seperate dateframe for z (Cr/Fe ratios), x (R-position), y (Z-position)
    #please note that here x is cylindrical R-coordinates and y is cylindrical Z-coordinates of particles.
    Cr_to_Fe = df['X_Cr_to_Fe']
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
        #Ni_est, Ni_bias, Ni_err, Ni_interval = JK_stats(clump_data["X_Ni"])
        #Mn_est, Mn_bias, Mn_err, Mn_interval = JK_stats(clump_data["X_Mn"])
        Cr_est, Cr_bias, Cr_err, Cr_interval = JK_stats(clump_data["X_Cr"])
        #Ti_est, Ti_bias, Ti_err, Ti_interval = JK_stats(clump_data["X_Ti"])

        #print out the ratios of our Jack-knife analysis. such as ratios of mean, error and confidence intervals.
        print("Est. :" , Cr_est/Fe_est)
        print("Error :", Cr_err/Fe_err)
        print("Conf. interval :", Cr_interval/Fe_interval)

        #print("Est. :" , Ni_est/Fe_est, Mn_est/Fe_est, Cr_est/Fe_est, Ti_est/Fe_est)
        #print("Error :" , Ni_err/Fe_err, Mn_err/Fe_err, Cr_err/Fe_err, Ti_err/Fe_err)
        #print("Conf. interval :" , Ni_interval/Fe_interval, Mn_interval/Fe_interval, Cr_interval/Fe_interval, Ti_interval/Fe_interval)

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
        clump_array.append([total_Fe,
                            total_Cr,
                            mean_Fe,
                            mean_Cr,
                            clump_Cr_to_Fe,
                            fst_pt_r,
                            fst_pt_z,
                            radius_dist])

    #create dataframe from the list clump_array since it is easier to work with.
    clumps_df = pd.DataFrame(clump_array,
                             columns = ['total_Fe',
                                        'total_Cr',
                                        'mean_Fe',
                                        'mean_Cr',
                                        'clump_Cr_to_Fe',
                                        'fst_pt_r',
                                        'fst_pt_z',
                                        'radius_dist'],
                            dtype = float)

    #sort it in descending order of clump_Cr_to_Fe column
    clumps_df.sort_values(by = ["clump_Cr_to_Fe"], ascending = False, inplace = True)

    #print the dataframe
    print(clumps_df.to_string())

    clumps_df = clumps_df.reset_index(drop = True)

    #plotting begins here.
    #cent is used for setting up domain size
    cent = 2.6e4

    #define figure and axes.
    fig, ax = plt.subplots(figsize = (5,10))

    size = df["X_Cr_to_Fe"]

    #plotting the scatter plot
    sp = ax.scatter(df['R_pos'], df['Z_pos'],
                    s = (60 * ((size - size.min())/(size.max() - size.min()))) + 0.5,
                    c = size, cmap = 'RdBu', norm = colors.LogNorm(vmax = 1.0e5, vmin = 1.0e-2))

    #few edits to the plot such as domain size, formtting of x and y axis, labels, and background color etc.
    ax.set_xlim(0,2*cent)
    ax.set_ylim(-2*cent,2*cent)
    ax.tick_params(labelsize = 16)
    yfmt = ScalarFormatterForceFormat()
    yfmt.set_powerlimits((0,0))
    ax.yaxis.set_major_formatter(yfmt)
    ax.xaxis.set_major_formatter(yfmt)
    ax.ticklabel_format(axis = 'both', style = 'sci', useMathText= True, scilimits = [0,0])
    ax.tick_params(axis='both',labelsize = 20)
    ax.xaxis.get_offset_text().set_fontsize(20)
    ax.yaxis.get_offset_text().set_fontsize(20)
    ax.set_xlabel('r (km)', fontsize = 20)
    ax.set_ylabel('z (km)', fontsize = 20)
    ax.set_facecolor('black')

    #draw the circle on the plot with coordinates of center point
    #of first element in clumps_df dataframe and radius.

    ax.add_patch(plt.Circle((clumps_df['fst_pt_r'].iloc[0], clumps_df['fst_pt_z'].iloc[0]),
                                   clumps_df['radius_dist'].iloc[0], fill = False, linewidth = 2.0,color ='white'))

    #draw colorbar below the plot and add labels to it.
    bar = ax.inset_axes([0.0,-0.15,1,0.02])
    c_bar = fig.colorbar(sp, cax = bar, orientation = 'horizontal', extend = 'min')
    c_bar.ax.tick_params(labelsize = 20)
    c_bar.set_label(label = r'$\mathtt{\frac{X_{Cr}}{X_{Fe}}}$', size = 26)

    #show the plot
    plt.show()

    #save figure
    fig.savefig("Scatter_plot_with_clump.png", bbox_inches = 'tight')
