#Script to create particles' Temperature and Density Trajectories 
#or History for both pure-def. and fully unbound remnant simulations. 

#Command to use for parallelization with fully unbound remnant
#-------> "mpirun -n <number of cores> python3 ./create_traj_parallel.py"

#Command to use for parallelization with pure-deflagration 
#-------> "mpirun -n <number of cores> python3 ./create_traj_parallel.py -puredef "

#Written by - V.Mehta (Nov. 2023)

#edit as per your need.


#importing the required libraries
from mpi4py import MPI
import pandas as pd
import numpy as np
import glob
import h5py
import argparse

#Define the fuction to read in hdf5 file and create the dataframe with particle 
#fields such as dens, temp, tags, press, etc.
#To read the field names in the same order as the columns in dataframe, uncomment i_r_p, df
#and add df to return variables. Print df when the fuction is called in the script. 

def get_data(file):
    f = h5py.File(file ,"r")
    dset = list(f.keys())
    r_s = f['real scalars']
    #i_r_p = f['particle names']
    #df = pd.DataFrame(data = i_r_p, dtype = str)
    t_p = f['tracer particles']
    data = pd.DataFrame(data = t_p)
    data.sort_values(by = [13], ascending = True, inplace = True)
    return data, r_s[1][1] # returns the dataframe of particle fields and time of the particle file.

#A code block to call '-puredef' flag while running the script.
#XXXXXXXXXX           DO NOT EDIT THESE LINES.          XXXXXXXXXX# 

parser = argparse.ArgumentParser()
parser.add_argument('-puredef', action='store_true', 
        help = 'Include -puredef flag after the script to turn on the puredef mode. Omit to assume fully unbound remnant.')
args = parser.parse_args()

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
n_proc = comm.Get_size()

#XXXXXXXXX -------------------------------------------- XXXXXXXXXX#

#Path to particle files in the directory. 
particle_files = glob.glob("/home/vmehta2/runs/3c397/DDT_OFFSET/ddt_offset_final/ddt_off_hdf5_part_*")

# If '-puredef' flag is define while running the script, following code block begins.
if args.puredef:
    
    #Print only for rank 0 processor.
    if rank == 0: 
        print("The -puredef switch is on. Initlaizing Binding Energy calculations for the last particle-file.")

    #Sort particle files in revere order and get the last particle file (first in this case).
    particle_files.sort(reverse = True)
    last_file = particle_files[0]
    
    #Calling the function to get the particle data and time.
    #Add 3rd variable if you return df in the fuction. That 3rd variable will be the field names.
    data_l, time_l = get_data(last_file)

    #Create individual dataframes for the fields we need.
    #These are required to calculate binding energy
    
    #density = data_l[:][1] #Density of particles.
    #pres = data_l[:][11]   #Pressre of particles.
    internal_e = data_l[:][2] #internal energy
    vel_x = data_l[:][15] #Velocity in x direction (r-direction for cylindrical co-ordinates)
    vel_y = data_l[:][16] #Velocity in y direction (z-direction for cylindircal co-ordinates)
    grav_pot = data_l[:][4] # Gravivatational potential
    part_tag = data_l[:][13] # Particle tags (very very important one)

    #Calculate the vel. magnitude
    vel_magnitude_square = np.square(vel_x) + np.square(vel_y)
    #Calculate Specific Binding Energy of the particles and crate seperate dataframe.
    binding_energy = (0.5*(vel_magnitude_square)) + internal_e + grav_pot
    binding_energy = pd.DataFrame(binding_energy)
    binding_energy = binding_energy.join(part_tag) #join two Dataframes (sp. bind. energy and p_tags) 
    
    #filter out the particles with pos. sp. bneg  and neg. sp. bneg 
    positive_binding_energy = binding_energy[binding_energy[0] > 0]
    negative_binding_energy = binding_energy[binding_energy[0] <= 0]
    
    if rank == 0:
        print("The simulation has a bound remnent left. ")
        print("The number of bound particles are: ", len(negative_binding_energy))
        print("The number of unbound particles are: ", len(positive_binding_energy))
    
    # Particle tags of unbound particles.
    traj_tag = positive_binding_energy[13].to_numpy()
    #if rank == 0:
        #print("The unbound particle tags are: ", traj_tag.tolist())

#  '-puredef' block of code is finished

# The default case  without flag '-puredef'
else:
    if rank == 0:
        print("The -puredef switch is off. Verifying if any particles are lost.")
    # Sometimes we lose particles in simulations that goes off domain. So we need to 
    # identify the lost particle tags. 

    particle_files.sort(reverse = False)
    
    # Read the last particle file by calling the fuction and get particle tags.
    last_file = particle_files[-1]
    data_l, time_l = get_data(last_file)
    l_ptags = data_l[:][13].to_numpy()
    
    # Read the first particle file by calling the fuction and get particle tags. 
    first_file = particle_files[0]
    data_f, time_f = get_data(first_file)
    f_ptags = data_f[:][13].to_numpy()

    # If the number of particle in the last time-step and the first time-step are same then,
    # no particles are lost.
    if len(l_ptags) == len(f_ptags):
        if rank == 0:
            print("No Particles are lost.")
        # Our traj_tag will be same as first or last-time step's particle tags 
        traj_tag = l_ptags 
    
    # Otherwise case, when we lose particles.
    else:
        if rank == 0:
            print("Particles are lost.")
        lost_index = np.isin(f_ptags, l_ptags, invert = True)
        lost_particles = f_ptags[lost_index]
        if rank == 0:
            # Printing particle tags of the particles that we lost.
            print("Lost particles tags are :", lost_particles.tolist()) 
        # Omit the particles which were lost and just use those tags that we have at the end of the simulation.
        traj_tag = l_ptags

#r_s = f['real scalars']
#time = r_s[1][1]

# Sort particle files in ascending order.
particle_files.sort(reverse = False)

# some basic preparion to distribute particles among the cores we use. 

#!!!!!!!!!! -------------------------------------------------------------------------------!!!!!!!!!!!!#

#This is important part of the code to parallelization. 
if rank == 0:
    # Crate an array with 0 to N numbers for N+1 number of files 
    part_files = np.arange(len(particle_files))
    
    # Get average files per core and remaining files and distribute remaining files to 
    # first few cores. 
    avg_files, remainder = divmod(part_files.size, n_proc)
    count = [avg_files + 1 if p < remainder else avg_files for p in range(n_proc)]
    
    # Store how many files each core will get to process into an array.
    count = np.array(count)

    # Get the first file indices that each core will process. 
    # Critical to preserve the order when we collect the processed data from all cores.
    displ = [sum(count[:p]) for p in range(n_proc)]
    displ = np.array(displ)
    #print(count, displ)

else:
    # For the cores other then 0th core processor, we initialize similar arrays so that 
    # they do not take in any random arguments for those arrays.
    part_files = None
    count = np.zeros(n_proc, dtype = int) # initialize count array with same size as number of cores.
    displ = None


# Broadcast count array to all the processors.
comm.Bcast(count, root = 0)

# Create an empty array with integer dtype in each processor with size 
# based on number of files they are allocated to process.
chunks = np.empty(count[rank],dtype = int)

# Scatter or distribute the 'part_files' array from 0th core to all the other cores
# based on the important flags that are provided such as number of files to process(count),
# first file index(displ), and data dtype of those indices(MPI.DOUBLE). 
# These divided elements(file indices) are stored into empty array that we initialize as 'chunk'
comm.Scatterv([part_files, count, displ, MPI.DOUBLE],chunks,root=0)


# Initialize zero 2D - arrays with size (number of traj) x (size of chunks array) 
# in each processor to store the values of Time, Temperature and Density from the particle files.
time_arr = np.zeros([len(traj_tag), len(chunks)], dtype = float) 
temp_arr = np.zeros([len(traj_tag), len(chunks)], dtype = float)
dens_arr = np.zeros([len(traj_tag), len(chunks)], dtype = float)

#print(chunks)
# end of the code block that distribute the file indices among the cores.
# ----------------------------------------------------------------------------------------------#

# Loop begins to read the file indices from chunks.
for n, index in enumerate(chunks):
    file = particle_files[int(index)] # read the name of the file at index 
    #print(file)
    f = h5py.File(file, 'r')
    dataset = list(f.keys()) 
    t_p = f['tracer particles']
    data = pd.DataFrame(data = t_p)
    
    # Filtering step to remove particles that are not in our 'traj_tag' array.
    data = data[data[13].isin(traj_tag)]
    data.sort_values(by = [13], ascending = True, inplace = True)
    
    # r_s to read the time.
    r_s = f['real scalars']
    # Create 4 arrays - time, p_tags, dens. and temp.

    time = r_s[1][1] #* np.ones(len(traj_tag))
    particle_tag = data[13].to_numpy()
    dens = data[1].to_numpy()
    temp = data[14].to_numpy()
    
    # Another for loop to transfer the values from our four 1D-arrays to three 2D-arrays 
    # that we initialized in each core.
    for i in range(len(traj_tag)):
        #tag = int(particle_tag[i]) #
        time_arr[i][n] = time
        temp_arr[i][n] = float(temp[i])
        dens_arr[i][n] = float(dens[i])
    f.close()

#!!!!!!!!!!!!! ------------------------------------------------------------------------------------------ !!!!!!!!!!!!!!#

# Important part to gather or collect the data from all the processors

# Three seperate final 1D-arrays with size of (traj_tags * total number of p_files)
# that will store data from all processors and later write .dat files
time_0 = np.zeros([len(traj_tag) * len(particle_files)],dtype = float)
temp_0 = np.zeros([len(traj_tag) * len(particle_files)],dtype = float)
dens_0 = np.zeros([len(traj_tag) * len(particle_files)],dtype = float)

# Recieveing count is different from the originally sent count (Scatterv).
# For 'Scatterv()', we need to send only file indices, but now 
# We collect (number of particles * indices) calls each-time.
rec_count = len(traj_tag) * count

# Our displacemnt will also increase with factor of number of particles.
# Recieving displ is also (number of particles * displ) 
displ = [sum(count[:q]) for q in range(n_proc)]
displ = np.array(displ)
rec_displ = len(traj_tag) * displ

#print(count, rec_count)
#print(displ, rec_displ)

# Flatten-out 2D arrays to 1D arrays.
time_arr = time_arr.T.flatten()
temp_arr = temp_arr.T.flatten()
dens_arr = dens_arr.T.flatten()
#data = list(zip(time_arr, temp_arr, dens_arr)) 

#print(time_arr)

# Call for Gather or collect the data from each processor to 0th processor. 
# This call also get similar flags as 'Scatterv()' call but on recieving side (2nd arg).
# time data is sent and stored to time_arr array of all processors 
# to time_0 array on 0th processor. Same for temp_arr and dens_arr. 
comm.Gatherv(time_arr, [time_0, rec_count, rec_displ, MPI.DOUBLE],root = 0)
comm.Gatherv(temp_arr, [temp_0, rec_count, rec_displ, MPI.DOUBLE],root = 0)
comm.Gatherv(dens_arr, [dens_0, rec_count, rec_displ, MPI.DOUBLE],root = 0)

#!!!!!!!!!!!!!!--------------------------------------------------------------------------------!!!!!!!!!!!!!!!!!!!!#

# Last step is to reshape the gathered 1D array into 2D array with size (total file x part_tag) 
# and transpose the array.
if rank == 0:
    time_0 = time_0.flatten().reshape(len(particle_files), len(traj_tag)).T
    temp_0 = temp_0.flatten().reshape(len(particle_files), len(traj_tag)).T
    dens_0 = dens_0.flatten().reshape(len(particle_files), len(traj_tag)).T

    # Loop over traj_tags and save each particle's data as .dat file.
    for j in range(len(traj_tag)):
        tag = int(traj_tag[j])
        history = "tempdens" + str(tag) + ".dat"
        np.savetxt(history, np.c_[time_0[j], temp_0[j], dens_0[j]], fmt = '%.7e') 

else:
    None


MPI.Finalize
#print(time_arr)
#print(temp_arr)
#print(dens_arr)

#    for i in range(len(particle_tag)):
#        tag = int(particle_tag[i])
#        tag = str(tag)
#        print(tag)
#        with open("tempdens" + tag + ".dat", "a") as f:
#            print("%.7E" %time,"%.7E" %temp[i],"%.7E" %dens[i], file = f)
