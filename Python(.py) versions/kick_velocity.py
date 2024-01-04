import yt
import numpy as np
import pandas as pd
import glob

files = glob.glob("def_high_dens_hdf5_chk_000062")
files.sort()

for file in files:
    ds = yt.load(file)
    ad = ds.all_data()
#    for i in sorted(ds.field_list):
#        print(i)

    grav_potential = ad[("flash", "gpot")]
    internal_energy = ad[("flash", "eint")]
    nuclear_energy = ad[("flash", "enuc")]
    velocity_x = ad[("flash", "velx")]
    velocity_y = ad[("flash", "vely")]
    density = ad[("flash","dens")]
    cell_volume = ad[("flash", "cell_volume")]

    velocity_magnitude = (np.square(velocity_x) + np.square(velocity_y))
    binding_energy = (0.5 * velocity_magnitude) + internal_energy + grav_potential

    indices = np.where(binding_energy > 0)
    indices_2 = np.where(binding_energy < 0)

    ejecta_density = density[indices]
    cell_mass = np.array(density) * np.array(cell_volume)

    mass = np.sum(cell_mass)
#    ejecta_mass = cell_mass[indices].sum()
    ejecta_momentum = np.array(cell_mass[indices]) * np.array(np.sqrt(velocity_magnitude[indices]))

    kick_velocity = (ejecta_momentum.sum())/(cell_mass[indices_2].sum())

    print(kick_velocity)

    

