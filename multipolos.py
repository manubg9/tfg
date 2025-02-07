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





