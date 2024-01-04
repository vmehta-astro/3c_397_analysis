import numpy as np

file = "./def_high_dens.dat"

E_nuc = np.loadtxt(file, usecols = [10], unpack = True)

t = np.loadtxt(file, usecols = [0], unpack = True)
time_step = []
for i in range(len(t)):
    if i == 0:
        dt = t[i] - t[i]
    else:
        dt = t[i] - t[i-1]
    time_step.append(dt)
#print(len(time_step))
#print(len(E_nuc))
nuc_E = time_step * E_nuc
#print(dt)

total_E_nuc = np.sum(nuc_E)

print(total_E_nuc) 
