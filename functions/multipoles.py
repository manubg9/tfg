import numpy as np
import scipy.special as sp
from math import factorial


def n_nm(n, m):

    """
    Parameters

    n: int
        Order of the multipole
    m: int
        Degree of the multipole
    ----------
    Returns: complex

        Prefactor por the multipoles in spherical coordinates Eq. (2.26)
    """
    
    x = 1/(4*np.pi)
    y = (2*n+1)/(n*(n+1))
    z = factorial(n-m)/factorial(n+m)
    
    return 1j*np.sqrt(x*y*z)


def j_n(n, x):
    """
    Parameters
    
    n: int
        Order
    x: float
        Argument 
    ----------
    Returns: 

        Spherical Bessel function of the first kind of order n.
    """
    
    return sp.spherical_jn(n, x)

def j_n_diff(n, x):
    """
    Parameters

    n: int
       Order
    x: float
        Argument 
    Returns:
    
    Derivative of the spherical Bessel function of the first kind of order n.
    """
    
    return sp.spherical_jn(n, x, derivative=True)

def y_n(n, x):
    """
    Parameters

    n: int
        Order 
    
    x: float
        Argument 
    ----------
    Returns:

    Spherical Bessel function of the second kind of order n.
    """
    
    return sp.spherical_yn(n, x)

def y_n_diff(n, x):
    """
    Parameters

    n: int
        Order 
    
    x: float
        Argument   
    ----------
    Returns:
    
    Derivative of the spherical Bessel function of the second kind of order n.
    """
    
    return sp.spherical_yn(n, x, derivative=True)

def h_n(n, x):
    """
    Parameters
    
    n: int
        Order 
    x: float
        Argument of the spherical Hankel function  
    ----------
    Returns:

    Spherical Hankel function of order n. Eq. (A.2.1)
    """
    
    return j_n(n, x) + 1j*y_n(n,x)

def h_n_diff(n, x):
    """
    Parameters
    
    n: int
        Order 
    x: float or array
        Argument of the spherical Hankel function  
    ----------
    Returns:

        Derivative of the spherical Hankel function of order n.
    """
    
    return j_n_diff(n, x) + 1j*y_n_diff(n,x)

def pi_nm(theta, n, m):
    """
    Parameters
    
    theta: float or array
        polar angle
    n: int
        Order of the multipole
    m: int
        Degree of the multipole
    ----------
    Returns:
    
    Function pi_nm from Eq. (2.27)
    """
    return m * sp.lpmv(m, n, np.cos(theta)) / np.sin(theta)

def tau_nm(theta, n, m):
    """
    Parameters
    
    theta: float or array
        polar angle
    n: int
        Order of the multipole
    m: int
        Degree of the multipole
    ----------
    Returns: array
    
    Function tau_nm from Eq. (2.28)
    
    """
    sino = np.sin(theta)
    coso = np.cos(theta)
    
    pnm = sp.lpmv(m, n, coso)
    pnm1 = sp.lpmv(m, n + 1, coso)
    return ((-(n + 1)* coso * pnm + (n - m + 1) * pnm1)) / sino
            
def X_nm(theta, phi, n, m):
  
    
    """
    Parameters
    
    theta: float or array
        Polar angle in radians
    phi: float or array
        Azimuthal angle in radians
    n: int
        Order of the multipole
    m: int
        Degree of the multipole
    ----------
    Returns:
     Xr: array
         Radial component of the vector spherical harmonic
     Xt: array
         Polar component of the vector spherical harmonic
     Xp: array
         Azimuthal component of the vector spherical harmonic
         
        Eq. 2.25
    """
    Xr = np.zeros_like(theta, dtype = complex)
    Xt = n_nm(n, m) * 1j * pi_nm(theta, n, m) * np.exp(1j*m*phi)
    Xp = -n_nm(n, m) * tau_nm(theta, n, m) * np.exp(1j*m*phi)
    
    return Xr, Xt, Xp

def M_nm(k, r, theta, phi, n, m, Hankel = False):
    """
    Transverse Electric mode in multipolar expansion
    with spherical Bessel or Hankel functions Eq. (2.29)
    ----------
    Parameters:
    
    k: float
        Wave number in m^-1
    r: float or array
        Radial distance in meters
    theta: float or array
        Polar angle in radians
    phi: float or array
        azimuthal angle in radiasns
    n: int
        Order of the multipole
    m: int
        Degree of the multipole
    Hankel: bool
        If True, uses spherical Hankel functions, otherwise uses spherical Bessel functions
    ----------
    Returns:
    
    r_mnm: array
        Radial component of the multipole field
    t_mnm: array
        Polar component of the multipole field
    p_mnm: array
        Azimuthal component of the multipole field
    """
    X, Y, Z = X_nm(theta, phi, n, m)
     
    if Hankel == False:

        r_mnm = j_n(n, k*r) * X
        t_mnm = j_n(n, k*r) * Y
        p_mnm = j_n(n, k*r) * Z
        return r_mnm, t_mnm, p_mnm
    
    elif Hankel == True:
        
        r_mnm = h_n(n, k*r) * X
        t_mnm = h_n(n, k*r) * Y
        p_mnm = h_n(n, k*r) * Z
        return r_mnm, t_mnm, p_mnm
    


def N_nm(k, r, theta, phi, n, m, Hankel = False):
    """
    Transverse Magentic mode in multipolar expansion
    with spherical Bessel or Hankel functions Eq. (2.30)
    ----------
    Parameters:
    
    k: float
        Wave number in m^-1
    r: float or array
        Radial distance in meters
    theta: float or array
        Polar angle in radians
    phi: float or array
        Azimuthal angle in radians
    n: int
        Order of the multipole
    m: int
        Degree of the multipole
    Hankel: bool
        If True, uses spherical Hankel functions, otherwise uses spherical Bessel functions
    ----------
    Returns:
    
    Nr: array
        Radial component of the multipole field
    Nt: array
        Polar component of the multipole field
    p_mnm2: array
        Azimuthal component of the multipole field
    """
    
    Xr, Xt, Xp  = X_nm(theta, phi, n, m)
    
    if Hankel == False:
        
        z = j_n(n, k*r)
        dz = j_n_diff(n, k*r)
                
    elif Hankel == True:

        z = h_n(n, k*r)
        dz = h_n_diff(n, k*r)
        
    pre = dz + z/(k*r)
    
    Nr = 1j*np.sqrt(n*(n+1))*sp.sph_harm(m, n, phi, theta)*z/(k*r)
    Nt = pre*(-Xp)
    Np = pre*Xt
    
    return Nr, Nt, Np







