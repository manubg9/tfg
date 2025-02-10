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
def n_nm(n,m):
    return 1j*np.sqrt((2*n+1)*factorial(n-m)/
                      (4*np.pi*n*(n+1)*factorial(n+m)))

def pi_nm(theta,n,m):
    return (m*sp.lpmv(m,n,np.cos(theta)))/(np.sin(theta))

def tau_nm(theta, n, m):
    P_nm = sp.lpmv(m, n, np.cos(theta))
    P_nm1 = sp.lpmv(m, n+1, np.cos(theta))
    return ((n+1)*np.cos(theta)*P_nm - (n-m+1)*P_nm1)/(np.sin(theta))

def X_nm(theta, phi, n, m):
    theta_nm = n_nm(n,m)*1j*pi_nm(theta,n,m)*np.exp(1j*m*phi)
    phi_nm = -n_nm(n,m)*1j*tau_nm(theta,n,m)*np.exp(1j*m*phi)
    r_nm = np.zeros_like(theta_nm)
    return r_nm, theta_nm, phi_nm

def M_nm(k, r, theta, phi, n, m):
    X, Y, Z = X_nm(theta, phi, n, m)
    return sp.spherical_jn(n, k*r)*np.array([X, Y, Z])

def N_nm(k, r, theta, phi, n, m):
    # Initialize rnm with the correct shape
    rnm = np.zeros((3, *r.shape), dtype=complex)
    rnm[0] = 1j * np.sqrt(n * (n + 1)) * sp.sph_harm(m, n, theta, phi) * sp.spherical_jn(n, k * r) / (k * r)
    
    # Initialize arrays for rnm2 and xnm
    rnm2 = np.zeros((3, *r.shape), dtype=complex)
    xnm = np.zeros((3, *r.shape), dtype=complex)
    
    # Compute the conversion at each point
    for i in range(r.shape[0]):
        for j in range(r.shape[1]):
            sph_to_cart = sph_cart(theta[i, j], phi[i, j])
            cart_to_sph = cart_sph(theta[i, j], phi[i, j])
            
            rnm2[:, i, j] = sph_to_cart @ np.array([(sp.jvp(n, k * r[i, j], 1) + sp.spherical_jn(n, k * r[i, j]) / (k * r[i, j])), 0, 0])
            xnm[:, i, j] = sph_to_cart @ np.array(X_nm(theta[i, j], phi[i, j], n, m))
            
            cross_product = np.cross(rnm2[:, i, j], xnm[:, i, j])
            rnm[:, i, j] += cart_to_sph @ cross_product
    
    return rnm





