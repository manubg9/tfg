import multipolos as mp
import numpy as np
import scipy.integrate as integ
import scipy.special as sp
import matplotlib.pyplot as plt
from math import factorial 
import time
#%%


def eta_f(k, f):
    
    x = 1j*k*f
    
    return -x*np.exp(x)/(2*np.pi)
    

def gamma_nm(n,m):
    
    x = 1/(4*np.pi)
    y = (2*n+1)/(n*(n+1))
    z = factorial(n-m)/factorial(n+m)
    
    return np.sqrt(x*y*z)

def Ppl(x,l,p):
    
    a = (np.sqrt(2)*x)**(abs(l))
    b = sp.assoc_laguerre(2*x**2,p, abs(l))
    c = np.exp(-x**2)
    return a*b*c


def bsc_tflg(k, rho, phi, z, n, m, l, p, f, w, pol, NA):
    
    a_max = np.radians(NA)
    prefactor = -4*(np.pi**2)*eta_f(k,f)*gamma_nm(n,m)*(1j**(l+n-m+1))*np.exp(1j*(l-m)*phi)
    
    def integrand(alpha, idx):
        
        
        sina = np.sin(alpha)
        cosa = np.cos(alpha)
        x = f/w * sina
        px, py = pol
        sigma = k * rho *sina
        
        comm = sina * Ppl(x, l, p) * np.exp(1j*k*z*cosa)
        
        first = (px + 1j*py)*np.exp(-1j*phi)*sp.jn(l-m-1, sigma)
        second = (px - 1j*py)*np.exp(1j*phi)*sp.jn(l-m+1, sigma)
        comb1 = mp.pi_nm(alpha, n, m) - mp.tau_nm(alpha, n, m)
        comb2 = mp.pi_nm(alpha, n, m) + mp.tau_nm(alpha, n, m)
        
    
    
        iae = comm*(first*comb1  + second*comb2)
            
        iam = comm*(first*(-comb1) + second*comb2)
        
        if idx == "e": 
        
            return iae
    
        elif idx == "m":
        
            return iam
    
        
    ae = prefactor * integ.quad_vec(integrand, 0, a_max, args =("e"))[0]
    am = prefactor * integ.quad_vec(integrand, 0, a_max, args=("m"))[0]
    
    return ae, am
                         

#%%

"bessel beams"

def bsc_bessel(k, rho, phi, z, n, m, l , pol, NA):
    a_max = np.radians(NA)
    px, py = pol
    sigma = k*rho*np.sin(a_max)
    prefactor = -4*np.pi/(1+np.cos(a_max)) * gamma_nm(n,m)*(1j**(l+n-m+1))*np.exp(1j*(l-m)*phi)*np.exp(1j*k*np.cos(a_max)*z)
    one = (px+1j*py)*np.exp(-1j*phi)*sp.jn(l-m-1, sigma)
    two = (px-1j*py)*np.exp(1j*phi)*sp.jn(l-m+1, sigma)
    mn = mp.pi_nm(a_max, n, m) - mp.tau_nm(a_max, n, m)
    pl = mp.pi_nm(a_max, n, m) + mp.tau_nm(a_max, n, m)
    ae = prefactor*(one*mn + two*pl)
    am = prefactor*(one*(-mn) + two*pl)
    
    return ae, am


    


