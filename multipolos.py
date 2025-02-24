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
    
    return m*sp.lpmv(m,n,np.cos(theta))/np.sin(theta)

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
    r_mnm = sp.spherical_jn(n, k*r) * X
    t_mnm = sp.spherical_jn(n, k*r) * Y
    p_mnm = sp.spherical_jn(n, k*r) * Z
    return r_mnm, t_mnm, p_mnm

def N_nm(k, r, theta, phi, n, m):
    
    
    rnm1 = 1j*np.sqrt(n*(n+1))*sp.sph_harm(m,n,phi, theta)*sp.spherical_jn(n, k*r)/(k*r)
    rnm2 = sp.spherical_jn(n, k*r, derivative = True) + sp.spherical_jn(n, k*r)/(k*r)
    xnm = X_nm(theta, phi, n, m)
    tnm1, pnm1 = xnm[1], xnm[2]
    tnm2 = -rnm2*pnm1
    pnm2 = rnm2*tnm1
    
    return rnm1, tnm2, pnm2


def pi_nm_vec(theta,n,m):
    
    d = theta.ndim
    
    if d == 2:
        vec = np.ones_like(theta, dtype = object)
        sh_th = theta.shape

        for k in range(sh_th[0]):
        
            for l in range(sh_th[1]):

                P_nm = sp.lpmn(m,n,np.cos(theta[k][l]))[0]
                for i in range(m+1):
                    P_nm[i] = i*P_nm[i] / np.sin(theta[k][l])
            
                vec[k][l] = vec[k][l] * P_nm
    
        return vec
    
    
    elif d == 3:
        vec = np.ones_like(theta, dtype = object)
        sh_th = theta.shape

        for k in range(sh_th[0]):
        
            for l in range(sh_th[1]):
                
                for p in range(sh_th[2]):
                
                    P_nm = sp.lpmn(m,n,np.cos(theta[k][l][p]))[0]
                    for i in range(m+1):
                        P_nm[i] = i*P_nm[i] / np.sin(theta[k][l][p])
            
                    vec[k][l][p] = vec[k][l][p] * P_nm
    
        return vec
               
            
    

