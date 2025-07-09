import multipoles as mp
import numpy as np
import scipy.integrate as integ
import scipy.special as sp
from math import factorial 


"""
Auxiliar Functions

"""
def eta_f(k, f):
    
    """
    Parameters
    -------------
        k: float
            Beam's wavenumber in m^-1
        
        f: float
            focal distance of the lens in m
            
    Returns: complex
    -------------
      eta_f defined in Section 3.1     
    """
    
    x = 1j*k*f
    
    return -x*np.exp(x)/(2*np.pi)
    

def gamma_nm(n,m):
    
    """
    Parameters
    -------------
         n: int
             multipole order
        m: int
            multipole degree
    Returns: float
    -------------
    
    gamma_nm function defined in Section 3.1
    
    """
    x = 1/(4*np.pi)
    y = (2*n+1)/(n*(n+1))
    z = factorial(n-m)/factorial(n+m)
    
    return np.sqrt(x*y*z)

def Ppl(x,l,p):
    
    """
    Parameters
    -------------
    
        x: float 
            argument
        l: int
            topological charge
        p: int
            radial index
    
    Returns: float
    -------------
    
        P_p^l(x) function defined in Section 3.1
    """
    a = (np.sqrt(2)*x)**(abs(l))
    b = sp.assoc_laguerre(2*x**2,p, abs(l))
    c = np.exp(-x**2)
    return a*b*c

"""
Tightly-Focused Laguerre-Gaussian Beams
"""


def bsc_tflg(k, rho, phi, z, n, m, l, p, f, w, pol, NA):
    
    """
    Parameters
    -------------
        k: float
            beam's wavenumber
        rho: float
            meters
        phi: float
            radians
        z: float
            meters
                arbitrary point for the beams in cylindrical coordinates
        n: int
            multipole order
        m: int
            multipole degree
        l: int
            topological charge
        p: int
            radial index
        f: float
            focal distance in m
        w: float
            width radius in m
        pol: tuple, array of size 2
            polarization state (px,py)         
        NA: float
                numerical aperture or half cone angle in degrees
    Returns:
    -------------
    
    ae, am: complex
        BSCs for Tightly-Focused Laguerre-Gaussian beams Eq. (3.3)
    """
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
    

                    
"""
Bessel beams
"""



def bsc_bessel(k, rho, phi, z, n, m, l , pol, NA):
    
    """
    Parameters
    -------------
        k: float
            beam's wavenumber
        rho: float
            meters
        phi: float
            radians
        z: float
            meters
                arbitrary point for the beams in cylindrical coordinates
        n: int
            multipole order
        m: int
            multipole degree
        l: int
            topological charge
        pol: tuple, array of size 2
            polarization state (px,py)         
        NA: float
                numerical aperture or half cone angle in degrees
    Returns:
    -------------
    
    ae, am: complex
        BSCs for Bessel beams Eq. (3.4)
    """
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


"""
Cylindrical vector Beams BSCs
"""

def bsc_cyl(k, rho, phi, z, n, m, l, pol, NA):
    
    """
    Parameters
    -------------
        k: float
            beam's wavenumber
        rho: float
            meters
        phi: float
            radians
        z: float
            meters
                arbitrary point for the beams in cylindrical coordinates
        n: int
            multipole order
        m: int
            multipole degree
        l: int
            topological charge
        pol: tuple, array of size 2
            polarization state (px,py)         
        NA: float
                numerical aperture or half cone angle in degrees
    Returns:
    -------------
    
    ae, am: complex
        BSCs for Cylindrical Vector beams Eq. (3.5)
    """
    a0 = np.radians(NA)
    polr, polp = pol
    cosa0 = np.cos(a0)
    sina0 = np.sin(a0)
    sigma = k*rho*sina0
    tau0 = mp.tau_nm(a0, n, m)
    pi0 = mp.pi_nm(a0, n, m)
    frac = 8*np.pi/(1+cosa0)
    arg1 = 1j*(l-m)*phi
    exp1 = np.exp(arg1)
    arg2 = 1j*k*cosa0*z
    exp2 = np.exp(arg2)
    gamma = gamma_nm(n, m)
    bessel = sp.jn(l-m, sigma)
    comm = frac*gamma*(1j**(l+n-m))*exp1*exp2*bessel
    comb1 = polr*tau0 - 1j*polp*pi0
    comb2 = polr*pi0 - 1j*polp*tau0
    ae = comm*comb1
    am = comm*comb2  
        
    return ae, am


