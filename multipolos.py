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
    
    
    rnm1 = 1j*np.sqrt(n*(n+1))*sp.sph_harm(m,n,theta, phi)*sp.spherical_jn(n, k*r)/(k*r)
    rnm2 = sp.jvp(n, k*r) + sp.spherical_jn(n, k*r)/(k*r)
    xnm = X_nm(theta, phi, n, m)
    tnm1, pnm1 = xnm[1], xnm[2]
    tnm2 = -rnm2*pnm1
    pnm2 = rnm2*tnm1
    
    return rnm1, tnm2, pnm2
    





