# -*- coding: utf-8 -*-
"""
Created on Wed Jun 18 16:32:16 2025

@author: Usuario
"""

"""
test mie scattering
"""

import numpy as np
import matplotlib.pyplot as plt
import mie_coeff as mie
import beams as bsc
import multipolos as mp


"""
parameters
"""

rP = 200e-9
laser = 1064e-9
k = 2*np.pi/laser
nmax = 15
size = 250

"""
domain
"""

extent = np.linspace(-2.5*laser, 2.5*laser, size)

x1 = extent
y1 = x1
X1, Y1 = np.meshgrid(x1, y1)
Z1 = np.zeros_like(X1)
r1 = np.sqrt(X1**2  + Y1**2 + Z1**2)
theta1 = np.arccos(Z1/r1)
phi1 = np.arctan2(Y1,X1)

x2 = extent
z2 = x2
X2, Z2 = np.meshgrid(x2, z2)
Y2 = np.zeros_like(X2)
r2 = np.sqrt(X2**2  + Y2**2 + Z2**2)
theta2 = np.arccos(Z2/r2)
phi2 = np.arctan2(Y2,X2)

pol0 = (1,0)
pol1 = (1,1j)/np.sqrt(2)
pol2 = (1,-1j)/np.sqrt(2)
pol3 = (0,1)
pol4 = (1,1)
pol5 = (1,-1)
#%%

"""
scattered fields
"""

Erz= np.zeros_like(X1, dtype = complex) 
Etz = np.zeros_like(X1, dtype = complex)
Epz = np.zeros_like(X1, dtype = complex)

Ery= np.zeros_like(X2, dtype = complex)
Ety = np.zeros_like(X2, dtype = complex)
Epy = np.zeros_like(X2, dtype = complex)

for i in range(1, nmax+1):
    for j in range(-i,i+1):
        print((i,j))
        Mrz, Mtz, Mpz = mp.M_nm(k, r1, theta1, phi1, i, j)
        Mry, Mty, Mpy = mp.M_nm(k, r2, theta2, phi2, i, j)
        Nrz, Ntz, Npz = mp.N_nm(k, r1, theta1, phi1, i, j)
        Nry, Nty, Npy = mp.N_nm(k, r2, theta2, phi2, i, j)
        
        ae, am = bsc.bsc_cyl(k, 0, 0, 0, i, j, 0, pol3, 30)
        
        an, bn = mie.mie_ab(i, laser, rP, 4)
        
        Erz += ae*an*Nrz + am*bn*Mrz
        Etz += ae*an*Ntz + am*bn*Mtz
        Epz += ae*an*Npz + am*bn*Mpz
        Ery += ae*an*Nry + am*bn*Mry
        Ety += ae*an*Nty + am*bn*Mty
        Epy += ae*an*Npy + am*bn*Mpy
        
        
E2z = np.abs(Erz)**2 + np.abs(Etz)**2 + np.abs(Epz)**2
E2z = np.where(r1> 200e-9, E2z, np.nan)
E2z = E2z/np.nanmax(E2z)

E2y = np.abs(Ery)**2 + np.abs(Ety)**2 + np.abs(Epy)**2
E2y = np.where(r2> 200e-9, E2y, np.nan)
E2y = E2y/np.nanmax(E2y)

#%%

plt.figure(0)
plt.clf()
plt.pcolormesh(X1,Y1, E2z, cmap="jet")
plt.gca().set_aspect("equal")
particle = plt.Circle((0,0), radius = 200e-9, color = "gray")
plt.gca().add_patch(particle)
plt.colorbar()
plt.xticks([])
plt.yticks([])

plt.figure(1)
plt.clf()
plt.pcolormesh(X2, Z2, E2y, cmap ="jet")
plt.gca().set_aspect("equal")
particle = plt.Circle((0,0), radius = 200e-9, color = "gray")
plt.gca().add_patch(particle)
plt.xticks([])
plt.yticks([])
