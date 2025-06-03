#%%
import multipolos as mp
import mie_coeff as mie 
import numpy as np
import matplotlib.pyplot as plt


#%%

# Parameters

wavelength = 1064e-9
radius = 200e-9
m_particle = 3.5
n_medium = 1.0
k = 2*np.pi/wavelength

# Domains
# z = 0 

x1 = np.linspace(-2.5*wavelength, 2.5*wavelength,500)
# x1 = x1[np.sqrt(x1**2+x1**2) >= 200e-9]
y1 = x1

z1 = 0
X1, Y1 = np.meshgrid(x1,y1)

r1 = np.sqrt(X1**2 + Y1**2)
r1 = np.where(r1 >=radius, r1, np.nan)
theta1 = np.arccos(z1/r1)
phi1 = np.arctan2(Y1,X1)

# y = 0

x2 = x1
z2 = y1
y2 = 0

X2, Z2 = np.meshgrid(x2,z2)

r2 = np.sqrt(X2**2 + Z2**2  )
r2 = np.where(r2 >= radius, r2, np.nan)
theta2 = np.arccos(Z2/r2)
phi2 = np.arctan2(y2,X2)

#%%

# VSWFs

n, m = 1,1 

Mrz, Mtz, Mpz = mp.M_nm(k, r1, theta1, phi1, n, m)
Mry, Mty, Mpy = mp.M_nm(k, r2, theta2, phi2, n, m)

Nrz, Ntz, Npz = mp.N_nm(k, r1, theta1, phi1, n, m)
Nry, Nty, Npy = mp.N_nm(k, r2, theta2, phi2, n, m)

# Mie Coefficients

a_n, b_n = mie.mie_ab(n, wavelength, radius, m_particle)

# Bullets

aMrz, aMtz, aMpz = a_n*Mrz, a_n*Mtz, a_n*Mpz
ampaMz = np.sqrt(abs(aMrz)**2 + abs(aMtz)**2 + abs(aMpz)**2)
ampaMz = ampaMz/np.nanmax(ampaMz)
paMrz, paMtz, paMpz = np.angle(aMrz), np.angle(aMtz), np.angle(aMpz)
aNrz, aNtz, aNpz = a_n*Nrz, a_n*Ntz, a_n*Npz
ampaNz = np.sqrt(abs(aNrz)**2 + abs(aNtz)**2 + abs(aNpz)**2)
ampaNz = ampaNz/np.nanmax(ampaNz)
paNrz, paNtz, paNpz = np.angle(aNrz), np.angle(aNtz), np.angle(aNpz)

aMry, aMty, aMpy = a_n*Mry, a_n*Mty, a_n*Mpy
ampaMy = np.sqrt(abs(aMry)**2 + abs(aMty)**2 + abs(aMpy)**2)
ampaMy = ampaMy/np.nanmax(ampaMy)
paMry, paMty, paMpy = np.angle(aMry), np.angle(aMty), np.angle(aMpy)
aNry, aNty, aNpy = a_n*Nry, a_n*Nty, a_n*Npy
ampaNy = np.sqrt(abs(aNry)**2 + abs(aNty)**2 + abs(aNpy)**2)
ampaNy = ampaNy/np.nanmax(ampaNy)
paNry, paNty, paNpy = np.angle(aNry), np.angle(aNty), np.angle(aNpy)

bMrz, bMtz, bMpz = b_n*Mrz, b_n*Mtz, b_n*Mpz
ampbMz = np.sqrt(abs(bMrz)**2 + abs(bMtz)**2 + abs(bMpz)**2)
ampbMz = ampbMz/np.nanmax(ampbMz)
pbMrz, pbMtz, pbMpz = np.angle(bMrz), np.angle(bMtz), np.angle(bMpz)
bNrz, bNtz, bNpz = b_n*Nrz, b_n*Ntz, b_n*Npz
ampbNz = np.sqrt(abs(bNrz)**2 + abs(bNtz)**2 + abs(bNpz)**2)
ampbNz = ampbNz/np.nanmax(ampbNz)
pbNrz, pbNtz, pbNpz = np.angle(bNrz), np.angle(bNtz), np.angle(bNpz)

bMry, bMty, bMpy = b_n*Mry, b_n*Mty, b_n*Mpy
ampbMy = np.sqrt(abs(bMry)**2 + abs(bMty)**2 + abs(bMpy)**2)
ampbMy = ampbMy/np.nanmax(ampbMy)
pbMry, pbMty, pbMpy = np.angle(bMry), np.angle(bMty), np.angle(bMpy)
bNry, bNty, bNpy = b_n*Nry, b_n*Nty, b_n*Npy
ampbNy = np.sqrt(abs(bNry)**2 + abs(bNty)**2 + abs(bNpy)**2)
ampbMy = ampbNy/np.nanmax(ampbNy)
pbNry, pbNty, pbNpy = np.angle(bNry), np.angle(bNty), np.angle(bNpy)



#%%

def no_ticks():
    plt.xticks([])
    plt.yticks([])


   
plt.clf()


plt.suptitle(r"Bullets Phase for Transverse Electric modes with $b_n$ by components with ($n=m=1$) ")
plt.figtext(0.1,0.7, r"$z=0$", rotation = "vertical")
plt.figtext(0.1,0.2, r"$y=0$", rotation = "vertical")

    

plt.subplot(231)
plt.title(r"$M_r$")
plt.pcolormesh(X1,Y1, pbMrz, shading="auto", cmap = "plasma")
particle = plt.Circle((0,0), radius = 200e-9, color = "gray")
plt.gca().add_patch(particle)
ref = plt.Rectangle((800e-9, -2000e-9), 1064e-9, 100e-9, facecolor="black", edgecolor = "white")
plt.gca().add_patch(ref)
plt.text(1250e-9,-1800e-9, r"$\mathbf{\lambda}$")
no_ticks()
plt.gca().set_aspect("equal")


plt.subplot(232)
plt.title(r"$M_{\theta}$")
plt.pcolormesh(X1,Y1,pbMtz, shading="auto", cmap = "plasma")
particle = plt.Circle((0,0), radius = 200e-9, color = "gray")
plt.gca().add_patch(particle)
ref = plt.Rectangle((800e-9, -2000e-9), 1064e-9, 100e-9, facecolor="black", edgecolor = "white")
plt.gca().add_patch(ref)
plt.text(1250e-9,-1800e-9, r"$\mathbf{\lambda}$")
no_ticks()
plt.gca().set_aspect("equal")


plt.subplot(233)
plt.title(r"$M_{\phi}$")
plt.pcolormesh(X1,Y1,pbMpz, shading="auto", cmap = "plasma")
particle = plt.Circle((0,0), radius = 200e-9, color = "gray")
plt.gca().add_patch(particle)
ref = plt.Rectangle((800e-9, -2000e-9), 1064e-9, 100e-9, facecolor="black", edgecolor = "white")
plt.gca().add_patch(ref)
plt.text(1250e-9,-1800e-9, r"$\mathbf{\lambda}$")
no_ticks()
plt.gca().set_aspect("equal")


plt.subplot(234)
# plt.title(r"$$")
no_ticks()
plt.pcolormesh(X2,Z2,pbMry, shading="auto", cmap = "plasma")
particle = plt.Circle((0,0), radius = 200e-9, color = "gray")
plt.gca().add_patch(particle)
ref = plt.Rectangle((800e-9, -2000e-9), 1064e-9, 100e-9, facecolor="black", edgecolor = "white")
plt.gca().add_patch(ref)
plt.text(1250e-9,-1800e-9, r"$\mathbf{\lambda}$")
plt.gca().set_aspect("equal")

plt.subplot(235)
# plt.title(r"$bnMnm$")
no_ticks()
plt.pcolormesh(X2,Z2,pbMty, shading="auto", cmap = "plasma")
particle = plt.Circle((0,0), radius = 200e-9, color = "gray")
plt.gca().add_patch(particle)
ref = plt.Rectangle((800e-9, -2000e-9), 1064e-9, 100e-9, facecolor="black", edgecolor = "white")
plt.gca().add_patch(ref)
plt.text(1250e-9,-1800e-9, r"$\mathbf{\lambda}$")
plt.gca().set_aspect("equal")

plt.subplot(236)
# plt.title(r"$bnMnm$")
no_ticks()
plt.pcolormesh(X2,Z2,pbMpy, shading="auto", cmap = "plasma" )
particle = plt.Circle((0,0), radius = 200e-9, color = "gray")
plt.gca().add_patch(particle)
ref = plt.Rectangle((800e-9, -2000e-9), 1064e-9, 100e-9, facecolor="black", edgecolor = "white")
plt.gca().add_patch(ref)
plt.text(1250e-9,-1800e-9, r"$\mathbf{\lambda}$")
plt.gca().set_aspect("equal")
plt.tight_layout()
plt.show()
plt.savefig("phase_bnMn1m1.png", dpi=500)

#%%

plt.clf()


plt.suptitle(r"Bullets Amplitude for Transverse Magnetics modes $n=m=1$ ")
plt.figtext(0.22,0.7, r"$z=0$", rotation = "vertical")
plt.figtext(0.22,0.2, r"$y=0$", rotation = "vertical")

    

plt.subplot(221)
plt.title(r"$anNnm$")
plt.pcolormesh(X1,Y1, ampaNz, shading="auto", cmap = "jet")
particle = plt.Circle((0,0), radius = 200e-9, color = "gray")
plt.gca().add_patch(particle)
ref = plt.Rectangle((800e-9, -2000e-9), 1064e-9, 100e-9, facecolor="black", edgecolor = "white")
plt.gca().add_patch(ref)
plt.text(1250e-9,-1800e-9, r"$\mathbf{\lambda}$")
no_ticks()
plt.gca().set_aspect("equal")


plt.subplot(222)
plt.title(r"$bnNnm$")
plt.pcolormesh(X1,Y1,ampbNz, shading="auto", cmap = "jet")
particle = plt.Circle((0,0), radius = 200e-9, color = "gray")
plt.gca().add_patch(particle)
ref = plt.Rectangle((800e-9, -2000e-9), 1064e-9, 100e-9, facecolor="black", edgecolor = "white")
plt.gca().add_patch(ref)
plt.text(1250e-9,-1800e-9, r"$\mathbf{\lambda}$")
no_ticks()
plt.gca().set_aspect("equal")


plt.subplot(223)
plt.pcolormesh(X2,Z2,ampaNy, shading="auto", cmap = "jet")
particle = plt.Circle((0,0), radius = 200e-9, color = "gray")
plt.gca().add_patch(particle)
ref = plt.Rectangle((800e-9, -2000e-9), 1064e-9, 100e-9, facecolor="black", edgecolor = "white")
plt.gca().add_patch(ref)
plt.text(1250e-9,-1800e-9, r"$\mathbf{\lambda}$")
no_ticks()
plt.gca().set_aspect("equal")


plt.subplot(224)
# plt.title(r"$$")
no_ticks()
plt.pcolormesh(X2,Z2,ampbNy, shading="auto", cmap = "jet")
particle = plt.Circle((0,0), radius = 200e-9, color = "gray")
plt.gca().add_patch(particle)
ref = plt.Rectangle((800e-9, -2000e-9), 1064e-9, 100e-9, facecolor="black", edgecolor = "white")
plt.gca().add_patch(ref)
plt.text(1250e-9,-1800e-9, r"$\mathbf{\lambda}$")
plt.gca().set_aspect("equal")

plt.tight_layout()
plt.show()
plt.savefig("amp_abNn1m1.png", dpi = 500)