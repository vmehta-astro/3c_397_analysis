####################################################################################################
# this is a script to plot the global quantities produced by the flash run. 
# it specifically made for 3C 397 but you can use it as per your accordance.
# Written by Vrutant Mehta - 2022
####################################################################################################



import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

#define your file path here
file = ("./def_high_dens.dat")

#define your quantites in this array, make sure you write them same as in your data file
quantities = ["#time",
              "mass", 
              "x-momentum",
              "y-momentum",
              #"z-momentum",  
              "E_internal+kinetic",
              "E_kinetic_(from_vel)",
              "E_internal",
              "E_grav",
              "E_restmass",
              "E_nuclear",
              "E_neutloss",
              "mass_burned",
              "mass_burned_to_NSQE",
              "mass_burned_to_NSE",
              "mass_burned_by_flame",
              "estimated_Ni56_mass",
              #"mass_with_dens_>_1e9",
              #"mass_with_dens_>_3e8",
              #"mass_with_dens_>_1e8",
              #"mass_with_dens_>_3e7",
              #"mass_with_dens_>_2e7",
              #"mass_with_dens_>_1.5e7",
              #"mass_with_dens_>_1e7",
              #"mass_with_dens_>_7e6",
              #"mass_with_dens_>_3e6",
              #"mass_with_dens_>_1e6",
              "burned_volume",
              "maximum_density",
              "minimum_flame_density",
              "T_max",
              "Pressure_at_T_max"]
#reading in the data from file and make Dataframe
global_data = pd.read_csv(file,
                          sep = "\s+",
                          header = 0,
                          usecols = quantities)
#another Dataframe which is sum of two columns
total_energy = global_data["E_internal+kinetic"] + global_data["E_grav"]
Energy_ratios = (global_data["E_kinetic_(from_vel)"])/(abs(global_data["E_grav"])+(global_data["E_internal"]))
#addition this another Dataframe into the original dataframe and quantities array
#no need to do this if you don't need this additional array
global_data.insert(9,"E_Total(Grav+internal+kinetic)", total_energy)
quantities.insert(9, "E_Total(Grav+internal+kinetic)")

global_data.insert(10, "E_ratios" + r'$(\frac{E_{kin}}{|E_{grav}| + E_{int}})$ ', Energy_ratios)
quantities.insert(10,"E_ratios" + r'$(\frac{E_{kin}}{|E_{grav}| + E_{int}})$ ')
#print(global_data)
print(np.sum(np.diff(global_data["#time"])*global_data["E_nuclear"][1:]))
#plotting figure
plt.figure(figsize=(24,40),facecolor="#FFFFFF")
plt.suptitle('High Density Deflagration, Central dens = 6.0e9 , bubble radius = 16 kms, offset = 10 kms',fontsize = 28, fontweight="bold")


#for loop for plotting each sub-plot
for i in range(len(quantities) - 1):
    plt.subplot(8, 3, i+1)
    plt.plot(global_data["#time"], global_data[quantities[i+1]])
    if (quantities[i+1] == "E_kinetic_(from_vel)" or
        quantities[i+1] == "E_nuclear" or
        quantities[i+1] == "E_internal+kinetic" or
        quantities[i+1] == "E_internal" or
        quantities[i+1] == "E_ratios" + r'$(\frac{E_{kin}}{|E_{grav}| + E_{int}})$ ' or
        quantities[i+1] == "mass_burned" or
        quantities[i+1] == "mass_burned_to_NSQE" or
        quantities[i+1] == "mass_burned_to_NSE" or
        quantities[i+1] == "estimated_Ni56_mass" or
        quantities[i+1] == "burned_volume" or
        quantities[i+1] == "T_max" or
        quantities[i+1] == "Pressure_at_T_max" or
        quantities[i+1] == "E_neutloss"):
        plt.yscale("log")
    
    plt.title(quantities[i+1])

plt.savefig("global_plots.png")



