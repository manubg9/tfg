# -*- coding: utMlf-8 -*-
"""
Created on Tue Feb  4 11:07:45 2025

@author: panch
"""

import monopolos as mnp
import numpy as np
import matplotlib.pyplot as plt

#%%

lambda_0 = 1
k0 = 2*np.pi/lambda_0

x1, y1, z1 = np.linspace(-1, 1, 100), np.linspace(-1, 1, 100), 0.01
X1, Y1 = np.meshgrid(x1, y1)
r1 = np.sqrt(X1**2 + Y1**2 + z1**2)
theta1 = np.arccos(z1/r1)
phi1 = np.arctan2(Y1, X1)


l0 = 1

M_lm_z0_l1 = []

for i in range(-l0,l0+1):
    MlmR, MlmT, MlmP = mnp.M_lm(k0, r1, theta1, phi1, l0, i)
    Magnitude = np.sqrt(np.abs(MlmR)**2 + np.abs(MlmT)**2 + np.abs(MlmP)**2)
    M_lm_z0_l1.append(Magnitude)

N_lm_z0_l1 = []

for i in range(-l0,l0+1):
    NlmR, NlmT, NlmP = mnp.N_lm(k0, r1, theta1, phi1, l0, i)
    Magnitude = np.sqrt(np.abs(NlmR)**2 + np.abs(NlmT)**2 + np.abs(NlmP)**2)
    N_lm_z0_l1.append(Magnitude)


x2 ,z2 = np.linspace(-1, 1, 100),  np.linspace(-1, 1, 100)
y2 = 0
X2, Z2 = np.meshgrid(x2, z2)

r2 = np.sqrt(X2**2 + y2**2 + Z2**2)
theta2 = np.arccos(Z2/r2)
phi2 = np.arctan2(y2, X2)

M_lm_y0_l1 = []

for i in range(-l0,l0+1):
    MlmR, MlmT, MlmP = mnp.M_lm(k0, r2, theta2, phi2, l0, i)
    Magnitude = np.sqrt(np.abs(MlmR)**2 + np.abs(MlmT)**2 + np.abs(MlmP)**2)
    M_lm_y0_l1.append(Magnitude)
 
N_lm_y0_l1 = []

for i in range(-l0,l0+1):
    NlmR, NlmT, NlmP = mnp.N_lm(k0, r2, theta2, phi2, l0, i)
    Magnitude = np.sqrt(np.abs(NlmR)**2 + np.abs(NlmT)**2 + np.abs(NlmP)**2)
    N_lm_y0_l1.append(Magnitude)


#%%
plt.clf()
plt.suptitle(r"$\left| M_m^l \right|$ and $\left| N_m^l \right|$ at $z=0$ with $l=1$")

plt.subplot(261)
plt.pcolormesh(X1, Y1, M_lm_z0_l1[0], shading="auto")
plt.colorbar()
plt.gca().set_aspect('equal')
plt.title(r"$m=-1$")
plt.xlabel(r"$x$")
plt.ylabel(r"$y$")

plt.subplot(262)

plt.pcolormesh(X1, Y1, M_lm_z0_l1[1], shading="auto")
plt.colorbar()
plt.gca().set_aspect('equal')
plt.title(r"$m=0$")
plt.xlabel(r"$x$")
plt.ylabel(r"$y$")


plt.subplot(263)

plt.pcolormesh(X1, Y1, M_lm_z0_l1[2], shading="auto")
plt.colorbar()
plt.gca().set_aspect('equal')
plt.title(r"$m=1$")
plt.xlabel(r"$x$")
plt.ylabel(r"$y$")


plt.subplot(264)

plt.pcolormesh(X1, Y1, N_lm_z0_l1[0], shading="auto", cmap = "hot")
plt.colorbar()
plt.gca().set_aspect('equal')
plt.title(r"$m=-1$")
plt.xlabel(r"$x$")
plt.ylabel(r"$y$")


plt.subplot(265)

plt.pcolormesh(X1, Y1, N_lm_z0_l1[1], shading="auto", cmap="hot")
plt.colorbar()
plt.gca().set_aspect('equal')
plt.title(r"$m=0$")
plt.xlabel(r"$x$")
plt.ylabel(r"$y$")


plt.subplot(266)

plt.pcolormesh(X1, Y1, N_lm_z0_l1[2], shading="auto", cmap = "hot")
plt.colorbar()
plt.gca().set_aspect('equal')
plt.title(r"$m=1$")
plt.xlabel(r"$x$")
plt.ylabel(r"$y$")

plt.subplot(267)
plt.pcolormesh(X2, Z2, M_lm_y0_l1[0], shading="auto")
plt.colorbar()
plt.gca().set_aspect('equal')
plt.title(r"$m=-1$")
plt.xlabel(r"$x$")
plt.ylabel(r"$z$")

plt.subplot(268)

plt.pcolormesh(X2, Z2, M_lm_y0_l1[1], shading="auto")
plt.colorbar()
plt.gca().set_aspect('equal')
plt.title(r"$m=0$")
plt.xlabel(r"$x$")
plt.ylabel(r"$z$")


plt.subplot(269)

plt.pcolormesh(X2, Z2, M_lm_y0_l1[2], shading="auto")
plt.colorbar()
plt.gca().set_aspect('equal')
plt.title(r"$m=1$")
plt.xlabel(r"$x$")
plt.ylabel(r"$z$")


plt.subplot(2,6,10)

plt.pcolormesh(X2, Z2, N_lm_y0_l1[0], shading="auto",cmap = "hot")
plt.text(100,100, "tetas")
plt.colorbar()
plt.gca().set_aspect('equal')
plt.title(r"$m=-1$")
plt.xlabel(r"$x$")
plt.ylabel(r"$z$")


plt.subplot(2,6,11)

plt.pcolormesh(X1, Z2, N_lm_y0_l1[1], shading="auto", cmap="hot")
plt.colorbar()
plt.gca().set_aspect('equal')
plt.title(r"$m=0$")
plt.xlabel(r"$x$")
plt.ylabel(r"$z$")


plt.subplot(2,6,12)

plt.pcolormesh(X2, Z2, N_lm_y0_l1[2], shading="auto", cmap = "hot")
plt.colorbar()
plt.gca().set_aspect('equal')
plt.title(r"$m=1$")
plt.xlabel(r"$x$")
plt.ylabel(r"$z$")


plt.tight_layout()




