import numpy as np
import scipy.special as sp
from math import factorial

def sph_cart(theta, phi):
    """
    Matrix for switching from spherical to cartesian coordinates:
    ----------
    Parameters:

    theta: float
        colatitude angle
    phi: float
        azimuthal angle 
    ----------
    Returns:

    Matrix
    """

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
    """
    Matrix for switching from cartesian to spherical coordinates:
    ----------
    Parameters:

    theta: float
        colatitude angle
    phi: float 
        azimuthal angle 
    ----------
    Returns:

    Matrix
    """

    sin_theta = np.sin(theta)
    cos_theta = np.cos(theta)
    sin_phi = np.sin(phi)
    cos_phi = np.cos(phi)

    return np.array([
        [sin_theta * cos_phi, sin_theta * sin_phi, cos_theta],
        [cos_theta * cos_phi, cos_theta * sin_phi, -sin_theta],
        [-sin_phi, cos_phi, 0]
    ])


def n_nm(n: int, m: int):

    """
    Parameters

    n: int
        Order of the multipole
    m: int
        Degree of the multipole
    ----------
    Returns:

    Prefactor por the multipoles in spherical coordinates in Beutel et al. 2023
    """
    
    a = (2*n + 1) / (4*np.pi * n * (n + 1))
    b = factorial(n-m)/factorial(n+m)
    return 1j * np.sqrt(a*b)


def j_n(n: int, x: float):
    """
    Parameters
    
    n: int
        Order of the multipole
    x: float
        Argumetent of the spherical Bessel function
    ----------
    Returns:

    Spherical Bessel function of the first kind of order n.
    """
    
    return sp.spherical_jn(n, x)

def j_n_diff(n: int, x: float):
    """
    Parameters

    n: int
       Order of the multipole
    x: float
        Argument of the spherical Bessel function  
    ----------
    Returns:
    
    Derivative of the spherical Bessel function of the first kind of order n.
    """
    
    return sp.spherical_jn(n, x, derivative=True)

def y_n(n: int, x: float):
    """
    Parameters

    n: int
        Order of the multipole
    
    x: float
        Argument of the spherical Bessel function  
    ----------
    Returns:

    Spherical Bessel function of the second kind of order n.
    """
    
    return sp.spherical_yn(n, x)

def y_n_diff(n: int, x: float):
    """
    Parameters

    n: int
        Order of the multipole
    
    x: float
        Argument of the spherical Bessel function  
    ----------
    Returns:
    
    Derivative of the spherical Bessel function of the second kind of order n.
    """
    
    return sp.spherical_yn(n, x, derivative=True)

def h_n(n: int, x: float):
    """
    Parameters
    
    n: int
        Order of the multipole
    x: float
        Argument of the spherical Hankel function  
    ----------
    Returns:

    Spherical Hankel function of order n.
    """
    
    return j_n(n, x) + 1j*y_n(n,x)

def h_n_diff(n: int, x: float):
    """
    Parameters
    
    n: int
        Order of the multipole
    x: float
        Argument of the spherical Hankel function  
    ----------
    Returns:

    Derivative of the spherical Hankel function of order n.
    """
    
    return j_n_diff(n, x) + 1j*y_n_diff(n,x)

def pi_nm(theta,n,m):
    """
    Parameters
    
    theta: float
        colatitude angle
    n: int
        Order of the multipole
    m: int
        Degree of the multipole
    ----------
    Returns:
    
    Function pi_nm from Beutel et al. 2023
    """
    return m * sp.lpmv(m, n, np.cos(theta)) / np.sin(theta)

def tau_nm(theta, n, m):
    """
    Parameters
    
    theta: float
        colatitude angle
    n: int
        Order of the multipole
    m: int
        Degree of the multipole
    ----------
    Returns:
    
    Function tau_nm from Beutel et al. 2023
    """

    sino = np.sin(theta)
    coso = np.cos(theta)
    
    pnm = sp.lpmv(m, n, coso)
    pnm1 = sp.lpmv(m, n + 1, coso)
    return ((-(n + 1)* coso * pnm + (n - m + 1) * pnm1)) / sino
            
def X_nm(theta, phi, n, m):
    """
    Parameters
    
    theta: float
        Polar angle
    phi: float
        azimuthal angle
    n: int
        Order of the multipole
    m: int
        Degree of the multipole
    ----------
    Returns:
    
    Normalized vector spherical harmonics in spherical coordinates
    """
    Xr = np.zeros_like(theta, dtype = complex)
    Xt = n_nm(n, m) * 1j * pi_nm(theta, n, m) * np.exp(1j*m*phi)
    Xp = -n_nm(n, m) * tau_nm(theta, n, m) * np.exp(1j*m*phi)
    
    return Xr, Xt, Xp

def M_nm(k, r, theta, phi, n, m, Hankel = False):
    """
    Function to calculate the Transverse Electric mode 
    in multipolar expansion with spherical Bessel or Hankel functions
    ----------
    Parameters:
    
    k: float
        Wave number
    r: float
        Radial distance
    theta: float
        Polar angle
    phi: float
        azimuthal angle
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
    Function to calculate the Transverse Magnetic mode 
    in multipolar expansion with spherical Bessel or Hankel functions
    ----------
    Parameters:
    
    k: float
        Wave number
    r: float
        Radial distance
    theta: float
        Polar angle
    phi: float
        azimuthal angle
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







