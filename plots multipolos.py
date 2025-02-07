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

x2 ,z2 = np.linspace(-1, 1, 100),  np.linspace(-1, 1, 100)
y2 = 0
X2, Z2 = np.meshgrid(x2, z2)

r2 = np.sqrt(X2**2 + y2**2 + Z2**2)
theta2 = np.arccos(Z2/r2)
phi2 = np.arctan2(y2, X2)

#%%
l1 = 1

M_lm_z0_l1 = []

for i in range(-l1,l1+1):
    MlmR, MlmT, MlmP = mnp.M_lm(k0, r1, theta1, phi1, l1, i)
    Magnitude = np.sqrt(np.abs(MlmR)**2 + np.abs(MlmT)**2 + np.abs(MlmP)**2)
    M_lm_z0_l1.append(Magnitude)

N_lm_z0_l1 = []

for i in range(-l1,l1+1):
    NlmR, NlmT, NlmP = mnp.N_lm(k0, r1, theta1, phi1, l1, i)
    Magnitude = np.sqrt(np.abs(NlmR)**2 + np.abs(NlmT)**2 + np.abs(NlmP)**2)
    N_lm_z0_l1.append(Magnitude)



M_lm_y0_l1 = []

for i in range(-l1,l1+1):
    MlmR, MlmT, MlmP = mnp.M_lm(k0, r2, theta2, phi2, l1, i)
    Magnitude = np.sqrt(np.abs(MlmR)**2 + np.abs(MlmT)**2 + np.abs(MlmP)**2)
    M_lm_y0_l1.append(Magnitude)
 
N_lm_y0_l1 = []

for i in range(-l1,l1+1):
    NlmR, NlmT, NlmP = mnp.N_lm(k0, r2, theta2, phi2, l1, i)
    Magnitude = np.sqrt(np.abs(NlmR)**2 + np.abs(NlmT)**2 + np.abs(NlmP)**2)
    N_lm_y0_l1.append(Magnitude)
    
#%%

l2 = 2


M_lm_z0_l2 = []

for i in range(-l2,l2+1):
    MlmR, MlmT, MlmP = mnp.M_lm(k0, r1, theta1, phi1, l2, i)
    Magnitude = np.sqrt(np.abs(MlmR)**2 + np.abs(MlmT)**2 + np.abs(MlmP)**2)
    M_lm_z0_l2.append(Magnitude)


N_lm_z0_l2 = []

for i in range(-l2,l2+1):
    NlmR, NlmT, NlmP = mnp.N_lm(k0, r1, theta1, phi1, l2, i)
    Magnitude = np.sqrt(np.abs(NlmR)**2 + np.abs(NlmT)**2 + np.abs(NlmP)**2)
    N_lm_z0_l2.append(Magnitude)


M_lm_y0_l2 = []

for i in range(-l2,l2+1):
    MlmR, MlmT, MlmP = mnp.M_lm(k0, r2, theta2, phi2, l2, i)
    Magnitude = np.sqrt(np.abs(MlmR)**2 + np.abs(MlmT)**2 + np.abs(MlmP)**2)
    M_lm_y0_l2.append(Magnitude)
 
N_lm_y0_l2 = []

for i in range(-l2,l2+1):
    NlmR, NlmT, NlmP = mnp.N_lm(k0, r2, theta2, phi2, l2, i)
    Magnitude = np.sqrt(np.abs(NlmR)**2 + np.abs(NlmT)**2 + np.abs(NlmP)**2)
    N_lm_y0_l2.append(Magnitude)

#%%

sym_M_N_z0 = [] 

for i in range(0, 2*l1 +1):
    
    print(i)

#%%
plt.figure(0)
plt.clf()
plt.suptitle(r"Electric and Magnetic Multipoles with $l=1$")

plt.subplot(261)
plt.pcolormesh(X1, Y1, M_lm_z0_l1[0], shading="auto")
plt.colorbar(shrink = 0.5)
plt.gca().set_aspect('equal')
plt.title(r"$m=-1$")
plt.xlabel(r"$x$")
plt.ylabel(r"$y$")

plt.subplot(262)

plt.pcolormesh(X1, Y1, M_lm_z0_l1[1], shading="auto")
plt.colorbar(shrink = 0.5)
plt.gca().set_aspect('equal')
plt.title(r"$m=0$")
plt.xlabel(r"$x$")
plt.ylabel(r"$y$")


plt.subplot(263)

plt.pcolormesh(X1, Y1, M_lm_z0_l1[2], shading="auto")
plt.colorbar(shrink = 0.5)
plt.gca().set_aspect('equal')
plt.title(r"$m=1$")
plt.xlabel(r"$x$")
plt.ylabel(r"$y$")


plt.subplot(264)

plt.pcolormesh(X1, Y1, N_lm_z0_l1[0], shading="auto", cmap = "hot")
plt.colorbar(shrink = 0.5)
plt.gca().set_aspect('equal')
plt.title(r"$m=-1$")
plt.xlabel(r"$x$")
plt.ylabel(r"$y$")


plt.subplot(265)

plt.pcolormesh(X1, Y1, N_lm_z0_l1[1], shading="auto", cmap="hot")
plt.colorbar(shrink = 0.5)
plt.gca().set_aspect('equal')
plt.title(r"$m=0$")
plt.xlabel(r"$x$")
plt.ylabel(r"$y$")


plt.subplot(266)

plt.pcolormesh(X1, Y1, N_lm_z0_l1[2], shading="auto", cmap = "hot")
plt.colorbar(shrink = 0.5)
plt.gca().set_aspect('equal')
plt.title(r"$m=1$")
plt.xlabel(r"$x$")
plt.ylabel(r"$y$")

plt.subplot(267)
plt.pcolormesh(X2, Z2, M_lm_y0_l1[0], shading="auto")
plt.colorbar(shrink = 0.5)
plt.gca().set_aspect('equal')
plt.title(r"$m=-1$")
plt.xlabel(r"$x$")
plt.ylabel(r"$z$")

plt.subplot(268)

plt.pcolormesh(X2, Z2, M_lm_y0_l1[1], shading="auto")
plt.colorbar(shrink = 0.5)
plt.gca().set_aspect('equal')
plt.title(r"$m=0$")
plt.xlabel(r"$x$")
plt.ylabel(r"$z$")


plt.subplot(269)

plt.pcolormesh(X2, Z2, M_lm_y0_l1[2], shading="auto")
plt.colorbar(shrink = 0.5)
plt.gca().set_aspect('equal')
plt.title(r"$m=1$")
plt.xlabel(r"$x$")
plt.ylabel(r"$z$")


plt.subplot(2,6,10)

plt.pcolormesh(X2, Z2, N_lm_y0_l1[0], shading="auto",cmap = "hot")
plt.colorbar(shrink = 0.5)
plt.gca().set_aspect('equal')
plt.title(r"$m=-1$")
plt.xlabel(r"$x$")
plt.ylabel(r"$z$")


plt.subplot(2,6,11)

plt.pcolormesh(X1, Z2, N_lm_y0_l1[1], shading="auto", cmap="hot")
plt.colorbar(shrink = 0.5)
plt.gca().set_aspect('equal')
plt.title(r"$m=0$")
plt.xlabel(r"$x$")
plt.ylabel(r"$z$")


plt.subplot(2,6,12)

plt.pcolormesh(X2, Z2, N_lm_y0_l1[2], shading="auto", cmap = "hot")
plt.colorbar(shrink = 0.5)
plt.gca().set_aspect('equal')
plt.title(r"$m=1$")
plt.xlabel(r"$x$")
plt.ylabel(r"$z$")


plt.figtext(0.30, 0.90, r"$\left| M_{lm} \right|$", fontsize = 20
            , weight = "heavy")

plt.figtext(0.70, 0.90, r"$\left| N_{lm} \right|$", fontsize = 20
            , weight = "heavy")

plt.savefig("Multipoles l1.png", dpi = 500)

#%%
plt.figure(1)
plt.clf()
plt.suptitle(r"Electric and Magnetic Multipoles with $l=2$")


plt.subplot(4,5,1)

plt.pcolormesh(X1, Y1, M_lm_z0_l2[0], shading="auto")
# plt.colorbar(shrink = 0.5)
plt.gca().set_aspect('equal')
plt.xlabel(r"$x$")
plt.ylabel(r"$y$")
plt.title(r"$m = -2$")


plt.subplot(4,5,2)

plt.pcolormesh(X1, Y1, M_lm_z0_l2[1], shading="auto")
# plt.colorbar(shrink = 0.5)
plt.gca().set_aspect('equal')
plt.xlabel(r"$x$")
plt.ylabel(r"$y$")
plt.title(r"$m = -1$")


plt.subplot(4,5,3)

plt.pcolormesh(X1, Y1, M_lm_z0_l2[2], shading="auto")
# plt.colorbar(shrink = 0.5)
plt.gca().set_aspect('equal')
plt.xlabel(r"$x$")
plt.ylabel(r"$y$")
plt.title(r"$m = 0$")


plt.subplot(4,5,4)

plt.pcolormesh(X1, Y1, M_lm_z0_l2[2], shading="auto")
# plt.colorbar(shrink = 0.5)
plt.gca().set_aspect('equal')
plt.xlabel(r"$x$")
plt.ylabel(r"$y$")
plt.title(r"$m = 1$")


plt.subplot(4,5,5)

plt.pcolormesh(X1, Y1, M_lm_z0_l2[3], shading="auto")
# plt.colorbar(shrink = 0.5)
plt.gca().set_aspect('equal')
plt.xlabel(r"$x$")
plt.ylabel(r"$y$")
plt.title(r"$m = 2$")

plt.subplot(4,5,11)

plt.pcolormesh(X1, Y1, N_lm_z0_l2[0], shading="auto", cmap = "hot")
# plt.colorbar(shrink = 0.5)
plt.gca().set_aspect('equal')
plt.xlabel(r"$x$")
plt.ylabel(r"$y$")
# plt.title(r"$m = -2$")


plt.subplot(4,5,12)

plt.pcolormesh(X1, Y1, N_lm_z0_l2[1], shading="auto", cmap = "hot")
# plt.colorbar(shrink = 0.5)
plt.gca().set_aspect('equal')
plt.xlabel(r"$x$")
plt.ylabel(r"$y$")
# plt.title(r"$m = -1$")


plt.subplot(4,5,13)

plt.pcolormesh(X1, Y1, N_lm_z0_l2[2], shading="auto", cmap = "hot")
# plt.colorbar(shrink = 0.5)
plt.gca().set_aspect('equal')
plt.xlabel(r"$x$")
plt.ylabel(r"$y$")
# plt.title(r"$m = 0$")


plt.subplot(4,5,14)

plt.pcolormesh(X1, Y1, N_lm_z0_l2[2], shading="auto", cmap = "hot")
# plt.colorbar(shrink = 0.5)
plt.gca().set_aspect('equal')
plt.xlabel(r"$x$")
plt.ylabel(r"$y$")
# plt.title(r"$m = 1$")


plt.subplot(4,5,15)

plt.pcolormesh(X1, Y1, N_lm_z0_l2[3], shading="auto", cmap = "hot")
# plt.colorbar(shrink = 0.5)
plt.gca().set_aspect('equal')
plt.xlabel(r"$x$")
plt.ylabel(r"$y$")
# plt.title(r"$m = 2$")

plt.subplot(4,5,6)

plt.pcolormesh(X2, Z2, M_lm_y0_l2[0], shading="auto")
# plt.colorbar(shrink = 0.5)
plt.gca().set_aspect('equal')
plt.xlabel(r"$x$")
plt.ylabel(r"$z$")
# plt.title(r"$m = -2$")


plt.subplot(4,5,7)

plt.pcolormesh(X2, Z2, M_lm_y0_l2[1], shading="auto")
# plt.colorbar(shrink = 0.5)
plt.gca().set_aspect('equal')
plt.xlabel(r"$x$")
plt.ylabel(r"$z$")
# plt.title(r"$m = -1$")


plt.subplot(4,5,8)

plt.pcolormesh(X2, Z2, M_lm_y0_l2[2], shading="auto")
# plt.colorbar(shrink = 0.5)
plt.gca().set_aspect('equal')
plt.xlabel(r"$x$")
plt.ylabel(r"$z$")
# plt.title(r"$m = 0$")


plt.subplot(4,5,9)

plt.pcolormesh(X2, Z2, M_lm_y0_l2[2], shading="auto")
# plt.colorbar(shrink = 0.5)
plt.gca().set_aspect('equal')
plt.xlabel(r"$x$")
plt.ylabel(r"$z$")
# plt.title(r"$m = 1$")


plt.subplot(4,5,10)

plt.pcolormesh(X2, Z2, M_lm_y0_l2[3], shading="auto")
# plt.colorbar(shrink = 0.5)
plt.gca().set_aspect('equal')
plt.xlabel(r"$x$")
plt.ylabel(r"$z$")
# plt.title(r"$m = 2$")

plt.subplot(4,5,16)

plt.pcolormesh(X2, Z2, N_lm_y0_l2[0], shading="auto", cmap = "hot")
# plt.colorbar(shrink = 0.5)
plt.gca().set_aspect('equal')
plt.xlabel(r"$x$")
plt.ylabel(r"$z$")
# plt.title(r"$m = -2$")


plt.subplot(4,5,17)

plt.pcolormesh(X2, Z2, N_lm_y0_l2[1], shading="auto", cmap = "hot")
# plt.colorbar(shrink = 0.5)
plt.gca().set_aspect('equal')
plt.xlabel(r"$x$")
plt.ylabel(r"$z$")
# plt.title(r"$m = -1$")


plt.subplot(4,5,18)

plt.pcolormesh(X2, Z2, N_lm_y0_l2[2], shading="auto", cmap = "hot")
# plt.colorbar(shrink = 0.5)
plt.gca().set_aspect('equal')
plt.xlabel(r"$x$")
plt.ylabel(r"$z$")
# plt.title(r"$m = 0$")


plt.subplot(4,5,19)

plt.pcolormesh(X2, Z2, N_lm_y0_l2[2], shading="auto", cmap = "hot")
# plt.colorbar(shrink = 0.5)
plt.gca().set_aspect('equal')
plt.xlabel(r"$x$")
plt.ylabel(r"$z$")
# plt.title(r"$m = 1$")


plt.subplot(4,5,20)

plt.pcolormesh(X2, Z2, N_lm_y0_l2[3], shading="auto", cmap = "hot")
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
plt.figtext(0.035, 0.70, r"$\left| M_{lm} \right|$", size = 20)
plt.figtext(0.035, 0.25, r"$\left| N_{lm} \right|$", size = 20)

plt.savefig("Multipolos l2.png", dpi = 500)



