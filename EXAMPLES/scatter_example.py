
"""


EXAMPLE SCRIPT FOR SCATTERED FIELDS


"""

import numpy as np
import matplotlib.pyplot as plt
import mie_coeff as mie
import beams as bsc
import multipoles as mp


"""
-----------
parameters
-----------
"""

a_p = 200e-9
n_p = 4
wavelength = 1064e-9
k = 2*np.pi/wavelength
nmax = 20
size = 250


f = 1e-3
w = 1e-3
l = 0
p = 0
r_p = (0,0,0)
rhop, phip, zp = r_p
NA = 30

"""
-------------------
Polarization States
--------------------
"""
pol0 = (1,0)
pol1 = (1,1j)/np.sqrt(2)
pol2 = (1,-1j)/np.sqrt(2)
pol3 = (0,1)
pol4 = (1,1)
pol5 = (1,-1)

"""
-----------
Domain
-----------
"""

extent = np.linspace(-0.7*wavelength, 0.7*wavelength, size)

# xy-plane z=0
x1 = extent
y1 = x1
X1, Y1 = np.meshgrid(x1, y1)
Z1 = np.zeros_like(X1)
r1 = np.sqrt(X1**2  + Y1**2 + Z1**2)
theta1 = np.arccos(Z1/r1)
phi1 = np.arctan2(Y1,X1)

# xz-plane y=0
x2 = extent
z2 = x2
X2, Z2 = np.meshgrid(x2, z2)
Y2 = np.zeros_like(X2)
r2 = np.sqrt(X2**2  + Y2**2 + Z2**2)
theta2 = np.arccos(Z2/r2)
phi2 = np.arctan2(Y2,X2)


#%%

"""
----------------------------------------------------------------------
Sum Eq. (4.6) to compute Normalized Scattered Fields Squared Amplitude
----------------------------------------------------------------------
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
        """
        choose your beam
        """
        # ae, am = bsc.bsc_tflg(k, rhop, phip, zp, i, j, l, p, f, w, pol0, NA)
        # ae, am = bsc.bsc_bessel(k, rhop, phip, zp, i, j, l, pol0, NA)
        ae, am = bsc.bsc_cyl(k, rhop, phip, zp, i, j, l, pol0, NA)
        
        if ae==0 and am==0:
            continue
        else:
            
            Mrz, Mtz, Mpz = mp.M_nm(k, r1, theta1, phi1, i, j, True)
            Mry, Mty, Mpy = mp.M_nm(k, r2, theta2, phi2, i, j, True)
            Nrz, Ntz, Npz = mp.N_nm(k, r1, theta1, phi1, i, j, True)
            Nry, Nty, Npy = mp.N_nm(k, r2, theta2, phi2, i, j, True)
        
       
            an, bn = mie.mie_ab(i, wavelength, a_p, n_p)
        
            Erz += ae*an*Nrz + am*bn*Mrz
            Etz += ae*an*Ntz + am*bn*Mtz
            Epz += ae*an*Npz + am*bn*Mpz
            Ery += ae*an*Nry + am*bn*Mry
            Ety += ae*an*Nty + am*bn*Mty
            Epy += ae*an*Npy + am*bn*Mpy
        
        
E2z = np.abs(Erz)**2 + np.abs(Etz)**2 + np.abs(Epz)**2
E2z = np.where(r1> a_p, E2z, np.nan)
E2z = E2z/np.nanmax(E2z)

E2y = np.abs(Ery)**2 + np.abs(Ety)**2 + np.abs(Epy)**2
E2y = np.where(r2> a_p, E2y, np.nan)
E2y = E2y/np.nanmax(E2y)

#%%
"""
---------------------------------------------------------------
Plotting 2D colormaps on $xy$-plane $z=0$ and $xz$-plane $y=0$
----------------------------------------------------------------
"""
plt.figure(0)
plt.clf()

plt.pcolormesh(X1,Y1, E2z, cmap="jet")
plt.gca().set_aspect("equal")
particle = plt.Circle((0,0), radius = a_p, color = "gray")
plt.gca().add_patch(particle)
ref= plt.Rectangle((-0.5e-6, -0.65e-6), wavelength, 0.05e-6, facecolor = "black", edgecolor = "white")
plt.gca().add_patch(ref)
plt.text(0, -0.55e-6, r"$\lambda_0$", color="white", fontsize="xx-large")
plt.colorbar()
plt.title(r"$xy$-plane $z=0$")
plt.xticks([])
plt.yticks([])
plt.show()

plt.figure(1)
plt.clf()
plt.pcolormesh(X2, Z2, E2y, cmap ="jet")
plt.gca().set_aspect("equal")
ref= plt.Rectangle((-0.5e-6, -0.65e-6), wavelength, 0.05e-6, facecolor = "black", edgecolor = "white")
plt.gca().add_patch(ref)
plt.text(0, -0.55e-6, r"$\lambda_0$", color="white", fontsize="xx-large")
particle = plt.Circle((0,0), radius = a_p, color = "gray")
plt.gca().add_patch(particle)
plt.title(r"$xz$-plane $y=0$")
plt.colorbar()
plt.xticks([])
plt.yticks([])
plt.show()