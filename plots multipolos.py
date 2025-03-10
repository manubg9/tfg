import multipolos as mnp
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
phi1 = np.where(phi1 < 0, phi1 + 2*np.pi, phi1)

x2 ,z2 = np.linspace(-1, 1, 100),  np.linspace(-1, 1, 100)
y2 = 0
X2, Z2 = np.meshgrid(x2, z2)

r2 = np.sqrt(X2**2 + y2**2 + Z2**2)
theta2 = np.arccos(Z2/r2)
phi2 = np.arctan2(y2, X2)
phi2 = np.where(phi2 < 0, phi2 + 2*np.pi, phi2)
#%%
n1 = 1

M_nm_z0_n1 = []
N_nm_z0_n1 = []
M_nm_y0_n1 = []
N_nm_y0_n1 = []

for i in range(-n1,n1+1):
    
    MnmR, MnmT, MnmP = mnp.M_nm(k0, r1, theta1, phi1, n1, i)
    Magnitude = np.sqrt(np.abs(MnmR)**2 + np.abs(MnmT)**2 + np.abs(MnmP)**2)
    M_nm_z0_n1.append(Magnitude)
    
    NnmR, NnmT, NnmP = mnp.N_nm(k0, r1, theta1, phi1, n1, i)
    Magnitude = np.sqrt(np.abs(NnmR)**2 + np.abs(NnmT)**2 + np.abs(NnmP)**2)
    N_nm_z0_n1.append(Magnitude)

    MnmR, MnmT, MnmP = mnp.M_nm(k0, r2, theta2, phi2, n1, i)
    Magnitude = np.sqrt(np.abs(MnmR)**2 + np.abs(MnmT)**2 + np.abs(MnmP)**2)
    M_nm_y0_n1.append(Magnitude)
 
    NnmR, NnmT, NnmP = mnp.N_nm(k0, r2, theta2, phi2, n1, i)
    Magnitude = np.sqrt(np.abs(NnmR)**2 + np.abs(NnmT)**2 + np.abs(NnmP)**2)
    N_nm_y0_n1.append(Magnitude)
    
#%%

n2 = 2

M_nm_z0_n2 = []
N_nm_z0_n2 = []
M_nm_y0_n2 = []
N_nm_y0_n2 = []

for i in range(-n2,n2+1):
    MnmR, MnmT, MnmP = mnp.M_nm(k0, r1, theta1, phi1, n2, i)
    Magnitude = np.sqrt(np.abs(MnmR)**2 + np.abs(MnmT)**2 + np.abs(MnmP)**2)
    M_nm_z0_n2.append(Magnitude)

    NnmR, NnmT, NnmP = mnp.N_nm(k0, r1, theta1, phi1, n2, i)
    Magnitude = np.sqrt(np.abs(NnmR)**2 + np.abs(NnmT)**2 + np.abs(NnmP)**2)
    N_nm_z0_n2.append(Magnitude)

    MnmR, MnmT, MnmP = mnp.M_nm(k0, r2, theta2, phi2, n2, i)
    Magnitude = np.sqrt(np.abs(MnmR)**2 + np.abs(MnmT)**2 + np.abs(MnmP)**2)
    M_nm_y0_n2.append(Magnitude)
 
    NnmR, NnmT, NnmP = mnp.N_nm(k0, r2, theta2, phi2, n2, i)
    Magnitude = np.sqrt(np.abs(NnmR)**2 + np.abs(NnmT)**2 + np.abs(NnmP)**2)
    N_nm_y0_n2.append(Magnitude)

maxN2z0 = np.max(N_nm_z0_n2)
#%%

sym_M_N_z0 = [] 
antisym_M_N_z0 = []

sym_M_N_y0 = []
antisym_M_N_y0 = []

for i in range(0, 2*n1 +1):
   
    a = (M_nm_z0_n1[i] + N_nm_z0_n1[i])/np.sqrt(2)
    b = (-M_nm_z0_n1[i] + N_nm_z0_n1[i])/np.sqrt(2)
    sym_M_N_z0.append(a)
    antisym_M_N_z0.append(b)
    
    c = (M_nm_y0_n1[i] + N_nm_y0_n1[i])/np.sqrt(2)
    d = (-M_nm_y0_n1[i] + N_nm_y0_n1[i])/np.sqrt(2)
    sym_M_N_y0.append(c)
    antisym_M_N_y0.append(d)
    
#%%
plt.figure(0)
plt.clf()
plt.suptitle(r"Electric and Magnetic Multipoles with $n=1$")

plt.subplot(261)
plt.pcolormesh(X1, Y1, M_nm_z0_n1[0], shading="auto")
plt.colorbar(shrink = 0.5)
plt.gca().set_aspect('equal')
plt.title(r"$m=-1$")
plt.xlabel(r"$x$")
plt.ylabel(r"$y$")

plt.subplot(262)

plt.pcolormesh(X1, Y1, M_nm_z0_n1[1], shading="auto")
plt.colorbar(shrink = 0.5)
plt.gca().set_aspect('equal')
plt.title(r"$m=0$")
plt.xlabel(r"$x$")
plt.ylabel(r"$y$")


plt.subplot(263)

plt.pcolormesh(X1, Y1, M_nm_z0_n1[2], shading="auto")
plt.colorbar(shrink = 0.5)
plt.gca().set_aspect('equal')
plt.title(r"$m=1$")
plt.xlabel(r"$x$")
plt.ylabel(r"$y$")


plt.subplot(264)

plt.pcolormesh(X1, Y1, N_nm_z0_n1[0], shading="auto", cmap = "hot")
plt.colorbar(shrink = 0.5)
plt.gca().set_aspect('equal')
plt.title(r"$m=-1$")
plt.xlabel(r"$x$")
plt.ylabel(r"$y$")


plt.subplot(265)

plt.pcolormesh(X1, Y1, N_nm_z0_n1[1], shading="auto", cmap="hot")
plt.colorbar(shrink = 0.5)
plt.gca().set_aspect('equal')
plt.title(r"$m=0$")
plt.xlabel(r"$x$")
plt.ylabel(r"$y$")


plt.subplot(266)

plt.pcolormesh(X1, Y1, N_nm_z0_n1[2], shading="auto", cmap = "hot")
plt.colorbar(shrink = 0.5)
plt.gca().set_aspect('equal')
plt.title(r"$m=1$")
plt.xlabel(r"$x$")
plt.ylabel(r"$y$")

plt.subplot(267)
plt.pcolormesh(X2, Z2, M_nm_y0_n1[0], shading="auto")
plt.colorbar(shrink = 0.5)
plt.gca().set_aspect('equal')
plt.title(r"$m=-1$")
plt.xlabel(r"$x$")
plt.ylabel(r"$z$")

plt.subplot(268)

plt.pcolormesh(X2, Z2, M_nm_y0_n1[1], shading="auto")
plt.colorbar(shrink = 0.5)
plt.gca().set_aspect('equal')
plt.title(r"$m=0$")
plt.xlabel(r"$x$")
plt.ylabel(r"$z$")


plt.subplot(269)

plt.pcolormesh(X2, Z2, M_nm_y0_n1[2], shading="auto")
plt.colorbar(shrink = 0.5)
plt.gca().set_aspect('equal')
plt.title(r"$m=1$")
plt.xlabel(r"$x$")
plt.ylabel(r"$z$")


plt.subplot(2,6,10)

plt.pcolormesh(X2, Z2, N_nm_y0_n1[0], shading="auto",cmap = "hot")
plt.colorbar(shrink = 0.5)
plt.gca().set_aspect('equal')
plt.title(r"$m=-1$")
plt.xlabel(r"$x$")
plt.ylabel(r"$z$")


plt.subplot(2,6,11)

plt.pcolormesh(X1, Z2, N_nm_y0_n1[1], shading="auto", cmap="hot")
plt.colorbar(shrink = 0.5)
plt.gca().set_aspect('equal')
plt.title(r"$m=0$")
plt.xlabel(r"$x$")
plt.ylabel(r"$z$")


plt.subplot(2,6,12)

plt.pcolormesh(X2, Z2, N_nm_y0_n1[2], shading="auto", cmap = "hot")
plt.colorbar(shrink = 0.5)
plt.gca().set_aspect('equal')
plt.title(r"$m=1$")
plt.xlabel(r"$x$")
plt.ylabel(r"$z$")


plt.figtext(0.30, 0.90, r"$\left| M_{nm} \right|$", fontsize = 20
            , weight = "heavy")

plt.figtext(0.70, 0.90, r"$\left| N_{nm} \right|$", fontsize = 20
            , weight = "heavy")

plt.subplots_adjust(top=0.949,
bottom=0.016,
left=0.041,
right=0.983,
hspace=0.0,
wspace=0.435)
plt.savefig("Multipoles n1.png", dpi = 500)

#%%
plt.figure(1)
plt.clf()
plt.suptitle(r"Electric and Magnetic Multipoles with $n=2$")


plt.subplot(4,5,1)

plt.pcolormesh(X1, Y1, M_nm_z0_n2[0], shading="auto")
# plt.colorbar(shrink = 0.5)
plt.gca().set_aspect('equal')
plt.xlabel(r"$x$") 
plt.ylabel(r"$y$")
plt.title(r"$m = -2$")


plt.subplot(4,5,2)

plt.pcolormesh(X1, Y1, M_nm_z0_n2[1], shading="auto")
# plt.colorbar(shrink = 0.5)
plt.gca().set_aspect('equal')
plt.xlabel(r"$x$")
plt.ylabel(r"$y$")
plt.title(r"$m = -1$")


plt.subplot(4,5,3)

plt.pcolormesh(X1, Y1, M_nm_z0_n2[2], shading="auto")
# plt.colorbar(shrink = 0.5)
plt.gca().set_aspect('equal')
plt.xlabel(r"$x$")
plt.ylabel(r"$y$")
plt.title(r"$m = 0$")


plt.subplot(4,5,4)

plt.pcolormesh(X1, Y1, M_nm_z0_n2[2], shading="auto")
# plt.colorbar(shrink = 0.5)
plt.gca().set_aspect('equal')
plt.xlabel(r"$x$")
plt.ylabel(r"$y$")
plt.title(r"$m = 1$")


plt.subplot(4,5,5)

plt.pcolormesh(X1, Y1, M_nm_z0_n2[3], shading="auto")
# plt.colorbar(shrink = 0.5)
plt.gca().set_aspect('equal')
plt.xlabel(r"$x$")
plt.ylabel(r"$y$")
plt.title(r"$m = 2$")

plt.subplot(4,5,11)

plt.pcolormesh(X1, Y1, N_nm_z0_n2[0], shading="auto", cmap = "hot")
# plt.colorbar(shrink = 0.5)
plt.gca().set_aspect('equal')
plt.xlabel(r"$x$")
plt.ylabel(r"$y$")
# plt.title(r"$m = -2$")


plt.subplot(4,5,12)

plt.pcolormesh(X1, Y1, N_nm_z0_n2[1], shading="auto", cmap = "hot")
# plt.colorbar(shrink = 0.5)
plt.gca().set_aspect('equal')
plt.xlabel(r"$x$")
plt.ylabel(r"$y$")
# plt.title(r"$m = -1$")


plt.subplot(4,5,13)

plt.pcolormesh(X1, Y1, N_nm_z0_n2[2], shading="auto", cmap = "hot")
# plt.colorbar(shrink = 0.5)
plt.gca().set_aspect('equal')
plt.xlabel(r"$x$")
plt.ylabel(r"$y$")
# plt.title(r"$m = 0$")


plt.subplot(4,5,14)

plt.pcolormesh(X1, Y1, N_nm_z0_n2[2], shading="auto", cmap = "hot")
# plt.colorbar(shrink = 0.5)
plt.gca().set_aspect('equal')
plt.xlabel(r"$x$")
plt.ylabel(r"$y$")
# plt.title(r"$m = 1$")


plt.subplot(4,5,15)

plt.pcolormesh(X1, Y1, N_nm_z0_n2[3], shading="auto", cmap = "hot")
# plt.colorbar(shrink = 0.5)
plt.gca().set_aspect('equal')
plt.xlabel(r"$x$")
plt.ylabel(r"$y$")
# plt.title(r"$m = 2$")

plt.subplot(4,5,6)

plt.pcolormesh(X2, Z2, M_nm_y0_n2[0], shading="auto")
# plt.colorbar(shrink = 0.5)
plt.gca().set_aspect('equal')
plt.xlabel(r"$x$")
plt.ylabel(r"$z$")
# plt.title(r"$m = -2$")


plt.subplot(4,5,7)

plt.pcolormesh(X2, Z2, M_nm_y0_n2[1], shading="auto")
# plt.colorbar(shrink = 0.5)
plt.gca().set_aspect('equal')
plt.xlabel(r"$x$")
plt.ylabel(r"$z$")
# plt.title(r"$m = -1$")


plt.subplot(4,5,8)

plt.pcolormesh(X2, Z2, M_nm_y0_n2[2], shading="auto")
# plt.colorbar(shrink = 0.5)
plt.gca().set_aspect('equal')
plt.xlabel(r"$x$")
plt.ylabel(r"$z$")
# plt.title(r"$m = 0$")


plt.subplot(4,5,9)

plt.pcolormesh(X2, Z2, M_nm_y0_n2[2], shading="auto")
# plt.colorbar(shrink = 0.5)
plt.gca().set_aspect('equal')
plt.xlabel(r"$x$")
plt.ylabel(r"$z$")
# plt.title(r"$m = 1$")


plt.subplot(4,5,10)

plt.pcolormesh(X2, Z2, M_nm_y0_n2[3], shading="auto")
# plt.colorbar(shrink = 0.5)
plt.gca().set_aspect('equal')
plt.xlabel(r"$x$")
plt.ylabel(r"$z$")
# plt.title(r"$m = 2$")

plt.subplot(4,5,16)

plt.pcolormesh(X2, Z2, N_nm_y0_n2[0], shading="auto", cmap = "hot")
# plt.colorbar(shrink = 0.5)
plt.gca().set_aspect('equal')
plt.xlabel(r"$x$")
plt.ylabel(r"$z$")
# plt.title(r"$m = -2$")


plt.subplot(4,5,17)

plt.pcolormesh(X2, Z2, N_nm_y0_n2[1], shading="auto", cmap = "hot")
# plt.colorbar(shrink = 0.5)
plt.gca().set_aspect('equal')
plt.xlabel(r"$x$")
plt.ylabel(r"$z$")
# plt.title(r"$m = -1$")


plt.subplot(4,5,18)

plt.pcolormesh(X2, Z2, N_nm_y0_n2[2], shading="auto", cmap = "hot")
# plt.colorbar(shrink = 0.5)
plt.gca().set_aspect('equal')
plt.xlabel(r"$x$")
plt.ylabel(r"$z$")
# plt.title(r"$m = 0$")


plt.subplot(4,5,19)

plt.pcolormesh(X2, Z2, N_nm_y0_n2[2], shading="auto", cmap = "hot")
# plt.colorbar(shrink = 0.5)
plt.gca().set_aspect('equal')
plt.xlabel(r"$x$")
plt.ylabel(r"$z$")
# plt.title(r"$m = 1$")


plt.subplot(4,5,20)

plt.pcolormesh(X2, Z2, N_nm_y0_n2[3], shading="auto", cmap = "hot")
# plt.colorbar(shrink = 0.5)
plt.gca().set_aspect('equal')
plt.xlabel(r"$x$")
plt.ylabel(r"$z$")
# plt.title(r"$m = 2$")

plt.tight_layout()
plt.subplots_adjust(top=0.927,
bottom=0.061,
left=0.087,
right=0.912,
hspace=0.394,
wspace=0.0)
plt.figtext(0.035, 0.70, r"$\left| M_{nm} \right|$", size = 20)
plt.figtext(0.035, 0.25, r"$\left| N_{nm} \right|$", size = 20)

plt.savefig("Multipolos n2.png", dpi = 500)


#%%

plt.figure(2)
plt.clf()
plt.suptitle(
    r"Symmetric and Antisymmetric linear combinations of Multipoles with $n=1$")


plt.subplot(261)

plt.pcolormesh(X1, Y1, sym_M_N_z0[0], shading = "auto")
# plt.colorbar()
plt.gca().set_aspect("equal")
plt.xlabel(r"$x$")
plt.ylabel(r"$y$")
plt.title(r"$m = -1$")


plt.subplot(262)

plt.pcolormesh(X1, Y1, sym_M_N_z0[1], shading = "auto")
# plt.colorbar()
plt.gca().set_aspect("equal")
plt.xlabel(r"$x$")
plt.ylabel(r"$y$")
plt.title(r"$m = 0$")


plt.subplot(263)

plt.pcolormesh(X1, Y1, sym_M_N_z0[2], shading = "auto")
# plt.colorbar()
plt.gca().set_aspect("equal")
plt.xlabel(r"$x$")
plt.ylabel(r"$y$")
plt.title(r"$m = 1$")


plt.subplot(264)

plt.pcolormesh(X1, Y1, antisym_M_N_z0[0], shading = "auto", cmap = "hot")
# plt.colorbar()
plt.gca().set_aspect("equal")
plt.xlabel(r"$x$")
plt.ylabel(r"$y$")
plt.title(r"$m=-1$")


plt.subplot(265)

plt.pcolormesh(X1, Y1, antisym_M_N_z0[1], shading = "auto", cmap = "hot")
# plt.colorbar()
plt.gca().set_aspect("equal")
plt.xlabel(r"$x$")
plt.ylabel(r"$y$")
plt.title(r"$m=0$")


plt.subplot(266)

plt.pcolormesh(X1, Y1, antisym_M_N_z0[2], shading = "auto", cmap = "hot")
# plt.colorbar()
plt.gca().set_aspect("equal")
plt.xlabel(r"$x$")
plt.ylabel(r"$y$")
plt.title(r"$m=1$")


plt.subplot(267)

plt.pcolormesh(X2, Z2, sym_M_N_y0[0], shading = "auto")
# plt.colorbar()
plt.gca().set_aspect("equal")
plt.xlabel(r"$x$")
plt.ylabel(r"$z$")
plt.title(r"$m = -1$")


plt.subplot(268)

plt.pcolormesh(X2, Z2, sym_M_N_y0[1], shading = "auto")
# plt.colorbar()
plt.gca().set_aspect("equal")
plt.xlabel(r"$x$")
plt.ylabel(r"$z$")
plt.title(r"$m = 0$")


plt.subplot(269)

plt.pcolormesh(X2, Z2, sym_M_N_y0[2], shading = "auto")
# plt.colorbar()
plt.gca().set_aspect("equal")
plt.xlabel(r"$x$")
plt.ylabel(r"$z$")
plt.title(r"$m = 1$")


plt.subplot(2,6,10)

plt.pcolormesh(X2, Z2, antisym_M_N_y0[0], shading = "auto", cmap = "hot")
# plt.colorbar()
plt.gca().set_aspect("equal")
plt.xlabel(r"$x$")
plt.ylabel(r"$z$")
plt.title(r"$m=-1$")


plt.subplot(2,6,11)

plt.pcolormesh(X2, Z2, antisym_M_N_y0[1], shading = "auto", cmap = "hot")
# plt.colorbar()
plt.gca().set_aspect("equal")
plt.xlabel(r"$x$")
plt.ylabel(r"$z$")
plt.title(r"$m=0$")


plt.subplot(2,6,12)

plt.pcolormesh(X2, Z2, antisym_M_N_y0[2], shading = "auto", cmap = "hot")
# plt.colorbar()
plt.gca().set_aspect("equal")
plt.xlabel(r"$x$")
plt.ylabel(r"$z$")
plt.title(r"$m=1$")

plt.tight_layout()
plt.subplots_adjust(top=0.949,
bottom=0.016,
left=0.041,
right=0.99,
hspace=0.0,
wspace=0.352)
plt.figtext(0.25, 0.90, 
        r"$\frac{\left| N_{nm} \right| + \left| M_{nm} \right|}{\sqrt{2}}$", 
        size = 20)

plt.figtext(0.75, 0.90, 
        r"$\frac{\left| N_{nm} \right| - \left| M_{nm} \right|}{\sqrt{2}}$", 
        size = 20)

plt.savefig("Multipoles linear comb.png", dpi = 500)