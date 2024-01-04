###########################################################################
#
# This files gives as an output the deviation of the velocities
# with respect to initial velcoity (velocity at t1)
# First, two files at t=t1 and t=t2 are provided as input
# Then all the required quantities are stored as dictionary with
# particle tag being the key.
#
# vel[0], vel[1] are velocity magnitude of file1 (t=t1) and file2(t=t2)
#
# By: Sudarshan Neopane, Oct 21 2020
#
#########################################################################
import h5py
import numpy as np
import matplotlib.pyplot as plt

# Name of the files at t=t1 and t=t2
fname = ["ddt_off_hdf5_part_047138",\
         "ddt_off_hdf5_part_048735"]
         
# Empty lists to store dictionaries for t=t1 and t=t2
vel = []
#posx = []
#posy = []
velx = []
vely = []
error = {0:"Deviation_velocitymagnitude"} # Stores the final devaition in dictionary
time = []

#file = file = h5py.File(fname[0],"r")
#print(file.keys())
#print(list(file["particle names"][14]))


# Loop over the files at t=t1 and t=t2
for i in range(len(fname)):
    # Give initial key to avois Ker Error
    vel.append({0:"Velocity_file%s"%(i)}) 
    velx.append({0:"Velocityx_file%s"%(i)})
    vely.append({0:"Velocityy_file%s"%(i)})
    #posx.append({0:"Positionx_file%s"%(i)})
    #posy.append({0:"Positiony_file%s"%(i)})
    # Open the file
    file = h5py.File(fname[i],"r")
    # Take the row and colum of the dataset
    row, column = file["tracer particles"].shape
    print(row)
    # Print time 
    time.append(float(file["real scalars"][1][1]))
    print(time)
    # Loop over the dataset and store required information in dictionary
    for j in range(int(row)):
        ptag = int(file["tracer particles"][j][11]) # Key of the dictionary
        velmag = np.sqrt(float(file["tracer particles"][j][13]) ** 2 \
                         + float(file["tracer particles"][j][14]) ** 2)
        vel[i].update({ptag : str(velmag)})
        velx[i].update({ptag : str(file["tracer particles"][j][13])})
        vely[i].update({ptag : str(file["tracer particles"][j][14])})
#        posx[i].update({ptag : str(file["tracer particles"][j][6])})
#        posy[i].update({ptag : str(file["tracer particles"][j][7])})

for key in vel[1]:
    if key != 0 :
        deviation = abs(float(vel[0][key]) - float(vel[1][key]))\
                    / float(vel[0][key])
        error.update({key : str(deviation)})
# Error for particle tag 20 can be accessed using error[20]
#print(error[20])

#print(velx[0][1])
#print(vely[0][1])

maxdev = 0
count1 = 0
count2 = 0
histo = []
for key in error:
    if key !=0 :
        histo.append(float(error[key])*100)
        if float(error[key]) > maxdev: maxdev = float(error[key])
        if float(error[key]) <= 0.01: count1 += 1
        if float(error[key]) > 0.01 and float(error[key]) <= 1: count2 += 1
print("Maximum deviation is:",maxdev)
print("Particles within 1% deviation is",count1)
print("Particles within 1% and 100% deviation is",count2)
plt.hist(histo,50)
plt.xlabel("Deviation in %")
plt.ylabel("Number of particles")
plt.title("DDT high density with offset:checked at t=%5.3fs and %5.3fs,"\
          "maxdev = %5.3f%%"\
          %(time[0],time[1], maxdev*100), fontsize = 10)
plt.savefig("histogram_"+fname[0][0:-17]+".png")
plt.show()
