)import numpy as np
import scipy.special as spec

def mie_ab(n, wavelength, radius, m_particle, n_medium=1.0):

    """
    Computes Mie coefficients a_n and b_n for a given order n.
    ----------
    Parameters:
    
        n          : int
            order of the Mie coefficients
        wavelength : float or array 
            in meters
        radius     : float
             particle radius in meters
        m_particle : complex or array
             complex refractive index of the particle
        n_medium   : float
            refractive index of the surrounding medium (default=1)
    ----------
    Returns:

        a_n, b_n : complex 
            Mie coefficients
    """
    k = 2 * np.pi * n_medium / wavelength
    x = k * radius
    m = m_particle / n_medium
    mx = m * x

    # Riccati-Bessel functions
    psi_n = lambda z: z * spec.spherical_jn(n, z)
    psi_n_prime = lambda z: spec.spherical_jn(n, z) + z * spec.spherical_jn(n, z, derivative=True)
    xi_n = lambda z: psi_n(z) + 1j * z * spec.spherical_yn(n, z)
    xi_n_prime = lambda z: psi_n_prime(z) + 1j * (spec.spherical_yn(n, z) + z * spec.spherical_yn(n, z, derivative=True))

    a_n = (m * psi_n(mx) * psi_n_prime(x) - psi_n(x) * psi_n_prime(mx)) / \
          (m * psi_n(mx) * xi_n_prime(x) - xi_n(x) * psi_n_prime(mx))

    b_n = (psi_n(mx) * psi_n_prime(x) - m * psi_n(x) * psi_n_prime(mx)) / \
          (psi_n(mx) * xi_n_prime(x) - m * xi_n(x) * psi_n_prime(mx))

    return a_n, b_n