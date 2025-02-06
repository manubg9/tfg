import numpy as np
import matplotlib.pyplot as plt
import scipy.special as sp
from math import factorial

def sph_cart(theta, phi):

    sin_theta = np.sin(theta)
    cos_theta = np.cos(theta)
    sin_phi = np.sin(phi)
    cos_phi = np.cos(phi)

    return np.array([
        [sin_theta * cos_phi, cos_theta * cos_phi, -sin_phi],
        [sin_theta * sin_phi, cos_theta * sin_phi, cos_phi],
        [cos_theta, -sin_theta, 0]
    ])

def cart_sph(theta, phi):

    sin_theta = np.sin(theta)
    cos_theta = np.cos(theta)
    sin_phi = np.sin(phi)
    cos_phi = np.cos(phi)

    return np.array([
        [sin_theta * cos_phi, sin_theta * sin_phi, cos_theta],
        [cos_theta * cos_phi, cos_theta * sin_phi, -sin_theta],
        [-sin_phi, cos_phi, 0]
    ])
def n_lm(l,m):
    return 1j*np.sqrt((2*l+1)*factorial(l-m)/
                      (4*np.pi*l*(l+1)*factorial(l+m)))

def pi_lm(theta,l,m):
    return (m*sp.lpmv(m,l,np.cos(theta)))/(np.sin(theta))

def tau_lm(theta, l, m):
    P_lm = sp.lpmv(m, l, np.cos(theta))
    P_lm1 = sp.lpmv(m, l+1, np.cos(theta))
    return ((l+1)*np.cos(theta)*P_lm - (l-m+1)*P_lm1)/(np.sin(theta))

def X_lm(theta, phi, l, m):
    theta_lm = n_lm(l,m)*1j*pi_lm(theta,l,m)*np.exp(1j*m*phi)
    phi_lm = -n_lm(l,m)*1j*tau_lm(theta,l,m)*np.exp(1j*m*phi)
    r_lm = np.zeros_like(theta_lm)
    return r_lm, theta_lm, phi_lm

def M_lm(k, r, theta, phi, l, m):
    X, Y, Z = X_lm(theta, phi, l, m)
    return sp.spherical_jn(l, k*r)*np.array([X, Y, Z])

def N_lm(k, r, theta, phi, l, m):
    # Initialize rlm with the correct shape
    rlm = np.zeros((3, *r.shape), dtype=complex)
    rlm[0] = 1j * np.sqrt(l * (l + 1)) * sp.sph_harm(m, l, theta, phi) * sp.spherical_jn(l, k * r) / (k * r)
    
    # Initialize arrays for rlm2 and xlm
    rlm2 = np.zeros((3, *r.shape), dtype=complex)
    xlm = np.zeros((3, *r.shape), dtype=complex)
    
    # Compute the conversion at each point
    for i in range(r.shape[0]):
        for j in range(r.shape[1]):
            sph_to_cart = sph_cart(theta[i, j], phi[i, j])
            cart_to_sph = cart_sph(theta[i, j], phi[i, j])
            
            rlm2[:, i, j] = sph_to_cart @ np.array([(sp.jvp(l, k * r[i, j], 1) + sp.spherical_jn(l, k * r[i, j]) / (k * r[i, j])), 0, 0])
            xlm[:, i, j] = sph_to_cart @ np.array(X_lm(theta[i, j], phi[i, j], l, m))
            
            cross_product = np.cross(rlm2[:, i, j], xlm[:, i, j])
            rlm[:, i, j] += cart_to_sph @ cross_product
    
    return rlm




#%%
"""
z = 0, calculations for Mlm and Nlm with l=m=1 and l=m=2

"""

lambda_0 = 1
k_0 = 2*np.pi/lambda_0

x1, y1, z1 = np.linspace(-1, 1, 100), np.linspace(-1, 1, 100), 0.01
X1, Y1 = np.meshgrid(x1, y1)
r1 = np.sqrt(X1**2 + Y1**2 + z1**2)
theta1 = np.arccos(z1/r1)
phi1 = np.arctan2(Y1, X1)

Mlm_R1_11, Mlm_T1_11, Mlm_P1_11 = M_lm(k_0, r1, theta1, phi1, 1, 1)
Mlm_magnitude1_11 = np.sqrt(np.abs(Mlm_R1_11)**2 + np.abs(Mlm_T1_11)**2 + np.abs(Mlm_P1_11)**2)

Mlm_R1_22, Mlm_T1_22, Mlm_P1_22 = M_lm(k_0, r1, theta1, phi1, 2, 2)
Mlm_magnitude1_22 = np.sqrt(np.abs(Mlm_R1_22)**2 + np.abs(Mlm_T1_22)**2 + np.abs(Mlm_P1_22)**2)

Nlm_R1_11, Nlm_T1_11, Nlm_P1_11 = N_lm(k_0, r1, theta1, phi1, 1, 1)
Nlm_magnitude1_11 = np.sqrt(np.abs(Nlm_R1_11)**2 + np.abs(Nlm_T1_11)**2 + np.abs(Nlm_P1_11)**2)

Nlm_R1_22, Nlm_T1_22, Nlm_P1_22 = N_lm(k_0, r1, theta1, phi1, 2, 2)
Nlm_magnitude1_22 = np.sqrt(np.abs(Nlm_R1_22)**2 + np.abs(Nlm_T1_22)**2 + np.abs(Nlm_P1_22)**2)

"""
y = 0, calculations for Mlm and Nlm with l=m=1 and l=m=2
 
"""

x2 ,z2 = np.linspace(-1, 1, 100),  np.linspace(-1, 1, 100)
y2 = 0
X2, Z2 = np.meshgrid(x2, z2)

r2 = np.sqrt(X2**2 + y2**2 + Z2**2)
theta2 = np.arccos(Z2/r2)
phi2 = np.arctan2(y2, X2)

Mlm_R2_11, Mlm_T2_11, Mlm_P2_11 = M_lm(k_0, r2, theta2, phi2, 1, 1)
Mlm_magnitude2_11 = np.sqrt(np.abs(Mlm_R2_11)**2 + np.abs(Mlm_T2_11)**2 + np.abs(Mlm_P2_11)**2)

Mlm_R2_22, Mlm_T2_22, Mlm_P2_22= M_lm(k_0, r2, theta2, phi2, 2, 2)
Mlm_magnitude2_22 = np.sqrt(np.abs(Mlm_R2_22)**2 + np.abs(Mlm_T2_22)**2 + np.abs(Mlm_P2_22)**2)

Nlm_R_y0_11, Nlm_T_y0_11, Nlm_P_y0_11 = N_lm(k_0, r2, theta2, phi2, 1, 1)
Nlm_magnitude_y0_11 = np.sqrt(np.abs(Nlm_R_y0_11)**2 + np.abs(Nlm_T_y0_11)**2 + np.abs(Nlm_P_y0_11)**2)

Nlm_R_y0_22, Nlm_T_y0_22, Nlm_P_y0_22 = N_lm(k_0, r2, theta2, phi2, 2, 2)
Nlm_magnitude_y0_22 = np.sqrt(np.abs(Nlm_R_y0_22)**2 + np.abs(Nlm_T_y0_22)**2 + np.abs(Nlm_P_y0_22)**2)

#%%

plt.clf()

"""
Plots
"""
plt.figure(0)
plt.suptitle(r"Electric and Magnetic Monopole Modules $|M_m^l|$, $|N_m^l|$")


plt.subplot(241)

plt.pcolormesh(X1, Y1, Mlm_magnitude1_11, shading='auto')
plt.gca().set_aspect('equal')
plt.xlabel(r"x")
plt.ylabel(r"y")
plt.title(r"$|M_m^l|$, $z=0$, $l=m=1$")
plt.colorbar()


plt.subplot(242)

plt.pcolormesh(X1, Y1, Mlm_magnitude1_22, shading='auto')
plt.gca().set_aspect('equal')
plt.xlabel(r"x")
plt.ylabel(r"y")
plt.title(r"$|M_m^l|$, $z=0$, $l=m=2$")
plt.colorbar()


plt.subplot(243)

plt.pcolormesh(X1, Y1, Nlm_magnitude1_11, shading='auto')
plt.gca().set_aspect('equal')
plt.xlabel(r"x")
plt.ylabel(r"y")
plt.title(r"$|N_m^l|$, $z=0$, $l=m=1$")
plt.colorbar()


plt.subplot(244)

plt.pcolormesh(X1, Y1, Nlm_magnitude1_22, shading='auto')
plt.gca().set_aspect('equal')
plt.xlabel(r"x")
plt.ylabel(r"y")
plt.title(r"$|N_m^l|$, $z=0$, $l=m=2$")
plt.colorbar()


plt.subplot(245)

plt.pcolormesh(X2, Z2, Mlm_magnitude2_11, shading='auto')
plt.gca().set_aspect('equal')
plt.xlabel(r"x")
plt.ylabel(r"z")
plt.title(r"$|M_m^l|$, $y=0$, $l=m=1$")
plt.colorbar()


plt.subplot(246)

plt.pcolormesh(X2, Z2, Mlm_magnitude2_22, shading = "auto")
plt.gca().set_aspect('equal')
plt.xlabel(r"x")
plt.ylabel(r"z")
plt.title(r"$|M_m^l|$, $y=0$, $l=m=2$")
plt.colorbar()


plt.subplot(247)

plt.pcolormesh(X2, Z2, Nlm_magnitude_y0_11, shading='auto')
plt.gca().set_aspect('equal')
plt.xlabel(r"x")
plt.ylabel(r"z")
plt.title(r"$|N_m^l|$, $y=0$, $l=m=1$")
plt.colorbar()


plt.subplot(248)

plt.pcolormesh(X2, Z2, Nlm_magnitude_y0_22, shading='auto')
plt.gca().set_aspect('equal')
plt.xlabel(r"x")
plt.ylabel(r"z")
plt.title(r"$|N_m^l|$, $y=0$, $l=m=2$")
plt.colorbar()



plt.savefig("monopolos.png", dpi = 500)