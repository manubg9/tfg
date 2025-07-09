"""
EXAMPLE SCRIPT FOR MULTIPOLES FIELDS
"""

import numpy as np
import matplotlib.pyplot as plt
import multipoles as mp


"""
-----------
Parameters
-----------
"""
r_p = 200e-9
wavelength = 1064e-9
k = 2*np.pi/wavelength
n = 1
m = 1
size = 250

"""
-------
Domain
-------
"""

extent = np.linspace(-2*wavelength, 2*wavelength, size)

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


extenth = np.linspace(-0.7*wavelength, 0.7*wavelength, size)

# xy-plane z=0
x1h = extenth
y1h = x1h
X1h, Y1h = np.meshgrid(x1h, y1h)
Z1h = np.zeros_like(X1h)
r1h = np.sqrt(X1h**2  + Y1h**2 + Z1h**2)
theta1h = np.arccos(Z1h/r1h)
phi1h = np.arctan2(Y1h,X1h)

# xz-plane y=0
x2h = extenth
z2h = x2h
X2h, Z2h = np.meshgrid(x2h, z2h)
Y2h = np.zeros_like(X2h)
r2h = np.sqrt(X2h**2  + Y2h**2 + Z2h**2)
theta2h = np.arccos(Z2h/r2h)
phi2h = np.arctan2(Y2h,X2h)


#%%

"""
---------------------------------------------------------------------------
Multipoles for Incident j=1 and Scatterd j=3 fields Eqs. (2.29) and (2.30)
--------------------------------------------------------------------------
"""


            
Mrz, Mtz, Mpz = mp.M_nm(k, r1, theta1, phi1, n, m)
Mza = np.abs(Mrz)**2 + np.abs(Mtz)**2 + np.abs(Mpz)**2
Mza = Mza/np.nanmax(Mza)
Mry, Mty, Mpy = mp.M_nm(k, r2, theta2, phi2, n, m)
Mya = np.abs(Mry)**2 + np.abs(Mty)**2 + np.abs(Mpy)**2
Mya = Mya/np.nanmax(Mya)
Nrz, Ntz, Npz = mp.N_nm(k, r1, theta1, phi1, n, m)
Nza = np.abs(Nrz)**2 + np.abs(Ntz)**2 + np.abs(Npz)**2
Nza = Nza/np.nanmax(Nza)
Nry, Nty, Npy = mp.N_nm(k, r2, theta2, phi2, n, m)
Nya = np.abs(Nry)**2 + np.abs(Nty)**2 + np.abs(Npy)**2
Nya = Nya/np.nanmax(Nya)

Mrzh, Mtzh, Mpzh = mp.M_nm(k, r1h, theta1h, phi1h, n, m, True)
Mzah = np.abs(Mrzh)**2 + np.abs(Mtzh)**2 + np.abs(Mpzh)**2
Mzah = np.where(r1h> r_p, Mzah, np.nan)
Mzah = Mzah/np.nanmax(Mzah)
Mryh, Mtyh, Mpyh = mp.M_nm(k, r2h, theta2h, phi2h, n, m, True)
Myah = np.abs(Mryh)**2 + np.abs(Mtyh)**2 + np.abs(Mpyh)**2
Myah = np.where(r2h> r_p, Myah, np.nan)
Myah = Myah/np.nanmax(Myah)
Nrzh, Ntzh, Npzh = mp.N_nm(k, r1h, theta1h, phi1h, n, m, True)
Nzah = np.abs(Nrzh)**2 + np.abs(Ntzh)**2 + np.abs(Npzh)**2
Nzah = np.where(r1h> r_p, Nzah, np.nan)
Nzah = Nzah/np.nanmax(Nzah)
Nryh, Ntyh, Npyh = mp.N_nm(k, r2h, theta2h, phi2h, n, m, True)
Nyah = np.abs(Nryh)**2 + np.abs(Ntyh)**2 + np.abs(Npyh)**2
Nyah = np.where(r2h> r_p, Nyah, np.nan)
Nyah = Nyah/np.nanmax(Nyah)


        



#%%
"""
---------------------------------------------------------------
Plotting 2D colormaps on $xy$-plane $z=0$ and $xz$-plane $y=0$
----------------------------------------------------------------
"""

plt.figure(0)
plt.clf()

plt.pcolormesh(X1,Y1, Mza, cmap="jet")
plt.gca().set_aspect("equal")
plt.colorbar()
plt.title(r"$M_{nm}^{(1)}$ $xy$-plane $z=0$")
ref= plt.Rectangle((0.6e-6, -1.6e-6), wavelength, 0.1e-6, facecolor = "black", edgecolor = "white")
plt.gca().add_patch(ref)
plt.text(1e-6, -1.4e-6, r"$\lambda_0$", color="black", fontsize="xx-large")
plt.xticks([])
plt.yticks([])
plt.show()

plt.figure(1)
plt.clf()
plt.pcolormesh(X2, Z2, Mya, cmap = "jet")
plt.gca().set_aspect("equal")
plt.title(r"$M_{nm}^{(1)}$ $xz$-plane $y=0$")
ref= plt.Rectangle((0.6e-6, -1.6e-6), wavelength, 0.1e-6, facecolor = "black", edgecolor = "white")
plt.gca().add_patch(ref)
plt.text(1e-6, -1.4e-6, r"$\lambda_0$", color="black", fontsize="xx-large")
plt.colorbar()
plt.xticks([])
plt.yticks([])
plt.show()

plt.figure(2)
plt.clf()

plt.pcolormesh(X1,Y1, Nza, cmap="jet")
plt.gca().set_aspect("equal")
plt.colorbar()
plt.title(r"$N_{nm}^{(1)}$ $xy$-plane $z=0$")
ref= plt.Rectangle((0.6e-6, -1.6e-6), wavelength, 0.1e-6, facecolor = "black", edgecolor = "white")
plt.gca().add_patch(ref)
plt.text(1e-6, -1.4e-6, r"$\lambda_0$", color="black", fontsize="xx-large")
plt.xticks([])
plt.yticks([])
plt.show()

plt.figure(3)
plt.clf()
plt.pcolormesh(X2, Z2, Nya, cmap = "jet")
plt.gca().set_aspect("equal")
plt.title(r"$N_{nm}^{(1)}$ $xz$-plane $y=0$")
ref= plt.Rectangle((0.6e-6, -1.6e-6), wavelength, 0.1e-6, facecolor = "black", edgecolor = "white")
plt.gca().add_patch(ref)
plt.text(1e-6, -1.4e-6, r"$\lambda_0$", color="black", fontsize="xx-large")
plt.colorbar()
plt.xticks([])
plt.yticks([])
plt.show()

plt.figure(4)
plt.clf()

plt.pcolormesh(X1h,Y1h, Mzah, cmap="jet")
plt.gca().set_aspect("equal")
plt.colorbar()
plt.title(r"$M_{nm}^{(3)}$ $xy$-plane $z=0$")
particle = plt.Circle((0,0), radius = r_p, color = "gray")
plt.gca().add_patch(particle)
ref= plt.Rectangle((-0.5e-6, -0.65e-6), wavelength, 0.05e-6, facecolor = "black", edgecolor = "white")
plt.gca().add_patch(ref)
plt.text(0, -0.55e-6, r"$\lambda_0$", color="white", fontsize="xx-large")
plt.xticks([])
plt.yticks([])
plt.show()

plt.figure(5)
plt.clf()
plt.pcolormesh(X2h, Z2h, Myah, cmap = "jet")
plt.gca().set_aspect("equal")
plt.title(r"$M_{nm}^{(3)}$ $xz$-plane $y=0$")
particle = plt.Circle((0,0), radius = r_p, color = "gray")
plt.gca().add_patch(particle)
ref= plt.Rectangle((-0.5e-6, -0.65e-6), wavelength, 0.05e-6, facecolor = "black", edgecolor = "white")
plt.gca().add_patch(ref)
plt.text(0, -0.55e-6, r"$\lambda_0$", color="white", fontsize="xx-large")
plt.colorbar()
plt.xticks([])
plt.yticks([])
plt.show()

plt.figure(6)
plt.clf()

plt.pcolormesh(X1h,Y1h, Nzah, cmap="jet")
plt.gca().set_aspect("equal")
plt.colorbar()
plt.title(r"$N_{nm}^{(3)}$ $xy$-plane $z=0$")
particle = plt.Circle((0,0), radius = r_p, color = "gray")
plt.gca().add_patch(particle)
ref= plt.Rectangle((-0.5e-6, -0.65e-6), wavelength, 0.05e-6, facecolor = "black", edgecolor = "white")
plt.gca().add_patch(ref)
plt.text(0, -0.55e-6, r"$\lambda_0$", color="white", fontsize="xx-large")
plt.xticks([])
plt.yticks([])
plt.show()

plt.figure(7)
plt.clf()
plt.pcolormesh(X2h, Z2h, Nyah, cmap = "jet")
plt.gca().set_aspect("equal")
plt.title(r"$N_{nm}^{(3)}$ $xz$-plane $y=0$")
particle = plt.Circle((0,0), radius = r_p, color = "gray")
plt.gca().add_patch(particle)
ref= plt.Rectangle((-0.5e-6, -0.65e-6), wavelength, 0.05e-6, facecolor = "black", edgecolor = "white")
plt.gca().add_patch(ref)
plt.text(0, -0.55e-6, r"$\lambda_0$", color="white", fontsize="xx-large")
plt.colorbar()
plt.xticks([])
plt.yticks([])
plt.show()


