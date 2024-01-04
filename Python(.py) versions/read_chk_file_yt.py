import yt 
from yt.units import dimensions
import unyt
import glob
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

plotting_data = []

#yt.enable_plugins()
M_sun = 1.988435e33 
cent = 2.0e9
files = glob.glob("def_high_dens_hdf5_chk_*")
files.sort()

#tag_file = pd.read_csv("p_tag.dat", sep = "\s+", usecols = [0], names = ["p_tag"])

#tags = tag_file["p_tag"]

for file in files:

    ds = yt.load(file)
    ad = ds.all_data()

    time = float(ds.current_time)
    #for i in sorted(ds.field_list):
    #    print(i)
    #for j in sorted(ds.derived_field_list):
    #    print(j)
    #p_tag = ad[("all","particle_tag")]
    #p_tag.sort()
    #print(p_tag[tags - 1])
    grav_potential = ad[("flash", "gpot")]
    internal_energy = ad[("flash", "eint")]
    nuclear_energy = ad[("flash", "enuc")]
    velocity_x = ad[("flash", "velx")]
    velocity_y = ad[("flash", "vely")]
    
    #flame_speed = ad[("flash", "fspd")]
    #flame_speed[flame_speed == 0.0] = np.nan
    #avg_fspd = np.nanmean(flame_speed)
    #avg_fspd = np.sum(flame_speed) / len(flame_speed)
#    print(flame_speed)
#    print(max(flame_speed), min(flame_speed))
#    print(avg_fspd/1.0e5)

    density = ad[("flash","dens")]
    cell_volume = ad[("flash", "cell_volume")]

    velocity_magnitude = (np.square(velocity_x) + np.square(velocity_y))

    binding_energy = (0.5 * velocity_magnitude) + internal_energy + grav_potential
    indices = np.where(binding_energy > 0)
    neg_indices = np.where(binding_energy < 0)
    ejecta_density = density[indices]
    cell_mass = np.array(density) * np.array(cell_volume)
    #cell_mass_weighted_E_nuc = cell_mass * np.array(nuclear_energy)
    
    mass = np.sum(cell_mass)
    ejecta_mass = cell_mass[indices].sum()
    ratio = (ejecta_mass/mass)
    bound_mass = mass - ejecta_mass
    bound_ratio = (mass - ejecta_mass)/mass

    plotting_data.append([time, ratio])

#    total_E_nuc = np.sum(cell_mass_weighted_E_nuc)
    #print('Ejecta Mass', "%5E" %ejecta_mass)
    print('Total Mass', "%5E" %mass)
    print('Total Mass in M_sol', mass/M_sun)
    print('Ejecta Mass', "%5E" %ejecta_mass)
    print('Ejecta Mass in M_sol', ejecta_mass/M_sun)
    print('Remnant Mass' "%5E" %bound_mass)
    print('Remnant Mass in M_sol', bound_mass/M_sun) 
    print('Ejecta mass/Total mass :', "%5E" %ratio)
    print()
    print()
#   with open("Ejecta_mass_and_E_nuc_data.dat", "a") as f:
#        print("%5E" %ejecta_mass, "%5E" %total_E_nuc, file = f) 

data_frame = pd.DataFrame(plotting_data, columns = ["time", "ratio"], dtype = float)

data_frame.to_csv('time_ratio.dat',sep = ' ',header = False, index = False, float_format = '%g')
#fig, ax = plt.subplots()
#ax.plot(data_frame["time"], data_frame["ratio"], ls = '-o-', c = 'red')
#ax.set_title("Ejecta Mass/Total Mass (High Central Density with Pure Deflagration)")
#ax.set_xlim(-0.2,6.2)
#ax.set_ylim(-0.2,0.3)
#ax.set_yscale('log')
#fig.savefig("ejecta_mass_to_total_mass_ratio.png", bbox_layout = 'tight')

    #mass_fraction = ejecta_mass / mass
    #print(len(binding_energy))
    #print(len(cell_mass))

    #print(indices)
    #print("mass: ", mass)
    #print("ejecta mass: ", ejecta_mass)
    #print("ejecta mass fraction: ", mass_fraction)


#    def binding_energy(field, data):
#        return(data["gas", "mass"] * ((0.5 * ((data[("flash", "velx")])**2 + (data[("flash", "vely")])**2)) + data[("flash", "eint")] + data[("flash", "gpot")])) 

#    ds.add_field(("flash", "bneg"), function = binding_energy, sampling_type = "cell", units = "auto")

    #print(ds.field_list)
#    positive_binding_energy = ad.cut_region(['obj["flash", "bneg"] > 1.0e37'])
#    negative_binding_energy = ad.cut_region(['obj["flash", "bneg"] < -1.0e37'])
#    pos_bn_eg_without_fluff = positive_binding_energy.cut_region(['obj["gas", "density"] > 1.0e-2'])
    
#    print("%.6E" %len(ad["flash", "bneg"]))
#    print("%.6E" %len(positive_binding_energy["flash", "bneg"]))
#    print("%.6E" %len(negative_binding_energy["flash", "bneg"]))

 
