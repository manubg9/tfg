# -*- coding: utf-8 -*-
"""
Created on Tue Jun 10 09:04:43 2025

@author: panch
"""

#%%
import beams as bsc
import numpy as np
import multipolos as mp
import matplotlib.pyplot as plt
import time

"""
uso lambda_0 = 1 um, w = f = 1mm, l = p = 0, (px,py) = (1,0), NA = 30º
  
"""
lb0 = 1e-6
k0 = 2*np.pi/lb0
w0 = 1e-3
f0 = 1e-3
l0 = 0
p0 = 0
pol0 = (1,0)
pol1 = (1/np.sqrt(2), 1j/np.sqrt(2))
pol2 = (0,1)
pol3 = (1,1)
pol4 = (1,-1)
NA0 = 30

# Dominio 

x1 = np.linspace(-2*lb0,2*lb0,100)
y1 = x1  
X1, Y1= np.meshgrid(x1,y1)
Z1 = np.zeros_like(X1)
# calculo cilindricas y esféricas para usar en los BSCs y los multipolos 
rho1 = np.sqrt(X1**2 + Y1**2)
rho1 = np.where(rho1==0, 1e-16, rho1)
phi1 = np.arctan2(Y1,X1)
phi1 = np.where(phi1 < 0, phi1 + 2*np.pi, phi1)
r1 = np.sqrt(rho1**2 + Z1**2)
theta1 = np.arccos(Z1/r1)

x2 = np.linspace(-2*lb0,2*lb0,100)
y2 = 0
z2 = x2  
X2, Z2= np.meshgrid(x2,z2)
Y2 = np.zeros_like(X2)

# calculo cilindricas y esféricas para usar en los BSCs y los multipolos 
rho2 = np.sqrt(X2**2 + Y2**2)
phi2= np.arctan2(Y2,X2)
phi2 = np.where(phi2 < 0, phi2 + 2*np.pi, phi2)
r2 = np.sqrt(rho2**2 + Z2**2)
r2 = np.where(r2 == 0, 1e-16, r2)
theta2 = np.arccos(Z2/r2)
#%%
# inicio los valores de E de forma que es más sencillo calcular el sumatorio

start = time.perf_counter()
E_rz = np.zeros_like(X1, dtype = complex)
E_thetaz = np.zeros_like(X1, dtype = complex)
E_phiz = np.zeros_like(X1, dtype = complex)
E_ry = np.zeros_like(X2, dtype = complex)
E_thetay = np.zeros_like(X2, dtype = complex)
E_phiy = np.zeros_like(X2, dtype = complex)


for i in range(1,51):
    for j in range(-i,i+1):

        print((i,j))
        # guardo las componentes en esféricas de de M y N
        Mrz, Mtz, Mpz = mp.M_nm(k0, r1, theta1, phi1, i, j)
        Nrz, Ntz, Npz = mp.N_nm(k0, r1, theta1, phi1, i, j)
        Mry, Mty, Mpy = mp.M_nm(k0, r2, theta2, phi2, i, j)
        Nry, Nty, Npy = mp.N_nm(k0, r2, theta2, phi2, i, j)
        # Calculo aenm y anmn
        # ae, am = bsc_tflg(k0, rho1, phi1, Z1, i, j, 1, p0, f0, w0, pol1, 30)
        # ae, am = bsc.bsc_tflg(k0, 0, 0, 0, i, j, l0, p0, f0, w0, pol0, 30)
        ae, am = bsc.bsc_bessel(k0, 0, 0, 0, i, j, l0, pol0, 70)
        # hago el sumatorio independiente de cada componente
        E_rz += ae*Nrz + am*Mrz
        E_thetaz += ae*Ntz + am*Mtz
        E_phiz += ae*Npz + am*Mpz
        E_ry += ae*Nry + am*Mry
        E_thetay += ae*Nty + am*Mty
        E_phiy += ae*Npy + am*Mpy

E_abs2z = (np.abs(E_rz)**2 + np.abs(E_thetaz)**2 + np.abs(E_phiz)**2)    
E_abs2z = E_abs2z/np.nanmax(E_abs2z)

E_abs2y = (np.abs(E_ry)**2 + np.abs(E_thetay)**2 + np.abs(E_phiy)**2)    
E_abs2y = E_abs2y/np.nanmax(E_abs2y)
    
end = time.perf_counter()

runtime = end - start 

print(runtime)

#%%
# testae = np.array([], dtype = float)
# testam = np.array([], dtype = float)
# pairs = np.array([], dtype = tuple)
# for i in range(1,86):
#     for j in range(-i, i+1):
#         pair = (i,j)
#         print(pair)
#         ae, am = bsc.bsc_tflg(k0, 0, 0, 0, i, j, 0, p0, f0, w0, pol0, 30)
#         absae = np.abs(ae)
#         absam = np.abs(am)
#         print(absae)
#         if absae < 1 and absae != 0 and absam < 1 and absam != 0:
            
#             testae = np.append(testae, absae)
#             testam = np.append(testam, absam)
#             pairs = np.append(pairs, pair)
# #%% 
# plt.clf()

# plt.pcolormesh(X1,Y1,E_abs2, cmap ="hot_r")        
# plt.gca().set_aspect("equal")
# plt.colorbar()
# plt.show()
#%%
    
plt.clf()
plt.pcolormesh(X1,Y1,E_abs2z, cmap ="hot_r")     
  
plt.gca().set_aspect("equal")
ref = plt.Rectangle((0.85e-6, -1.8e-6), 1e-6, 0.1e-6, facecolor="black", edgecolor = "white")
plt.text(1.3e-6, -1.6e-6, r"$\lambda_0$", fontsize = "large")
plt.gca().add_patch(ref)
plt.colorbar()
plt.xticks([])
plt.yticks([])
plt.show()

#%%
    
plt.clf()
plt.pcolormesh(X2,Z2,E_abs2y, cmap ="hot_r")     
  
plt.gca().set_aspect("equal")
ref = plt.Rectangle((0.85e-6, -1.8e-6), 1e-6, 0.1e-6, facecolor="black", edgecolor = "white")
plt.text(1.3e-6, -1.6e-6, r"$\lambda_0$", fontsize = "large")
plt.gca().add_patch(ref)
plt.colorbar()
plt.xticks([])
plt.yticks([])
plt.show()

#%%

# plt.clf()

# test = bsc_bessel(k0, rho1, phi1, z1, 11, 7, l0,pol0, NA0)[0] 
# plt.pcolormesh(X1,Y1,np.abs(test), cmap = "jet")
# plt.gca().set_aspect("equal")
# plt.colorbar()

