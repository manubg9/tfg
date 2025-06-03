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
#%%

"""
uso lambda_0 = 1 um, w = f = 1mm, l = p = 0, (px,py) = (1,0), NA = 30º
  
"""
lb0 = 1e-6
k0 = 2*np.pi/lb0
w0 = 1e-3
f0 = 1e-3
l0 = 0
p0 = 0
pol0 = (1,0)
pol1 = (1/np.sqrt(2), 1j/np.sqrt(2))
NA0 = 30

# Dominio 

x1 = np.linspace(-2*lb0,2*lb0,100)
y1 = x1  
X1, Y1= np.meshgrid(x1,y1)
Z1 = np.zeros_like(X1)
# calculo cilindricas y esféricas para usar en los BSCs y los multipolos 
rho1 = np.sqrt(X1**2 + Y1**2)
rho1 = np.where(rho1==0, 1e-16, rho1)
phi1 = np.arctan2(Y1,X1)
phi1 = np.where(phi1 < 0, phi1 + 2*np.pi, phi1)
r1 = np.sqrt(rho1**2 + Z1**2)
theta1 = np.arccos(Z1/r1)

x2 = np.linspace(-2*lb0,2*lb0,100)
y2 = 0
z2 = x2  
X2, Z2= np.meshgrid(x2,z2)
Y2 = np.zeros_like(X2)

# calculo cilindricas y esféricas para usar en los BSCs y los multipolos 
rho2 = np.sqrt(X2**2 + Y2**2)
phi2= np.arctan2(Y2,X2)
phi2 = np.where(phi2 < 0, phi2 + 2*np.pi, phi2)
r2 = np.sqrt(rho2**2 + Z2**2)
r2 = np.where(r2 == 0, 1e-16, r2)
theta2 = np.arccos(Z2/r2)
#%%
# inicio los valores de E de forma que es más sencillo calcular el sumatorio

start = time.perf_counter()
E_r = np.zeros_like(X1, dtype = complex)
E_theta = np.zeros_like(X1, dtype = complex)
E_phi = np.zeros_like(X1, dtype = complex)


for i in range(1,6):
    for j in range(-i,i+1):

        print((i,j))
        # guardo las componentes en esféricas de de M y N
        Mr, Mt, Mp = mp.M_nm(k0, r1, theta1, phi1, i, j)
        Nr, Nt, Np = mp.N_nm(k0, r1, theta1, phi1, i, j)
        # Calculo aenm y anmn
        ae, am = bsc_tflg(k0, rho1, phi1, Z1, i, j, 1, p0, f0, w0, pol1, 30)
        # ae, am = bsc_tflg(k0, rho2, phi2, Z2, i, j, l0, p0, f0, w0, pol0, 30)
        # ae, am = bsc_bessel(k0, rho1, phi1, Z1, i, j, l0, pol0, 30)
        # hago el sumatorio independiente de cada componente
        E_r += ae*Nr + am*Mr
        E_theta += ae*Nt + am*Mt 
        E_phi += ae*Np + am*Mp
        
sino = np.sin(theta1)
coso = np.cos(theta1)
sinp = np.sin(phi1)
cosp = np.cos(phi1)

E_x = E_r*sino*cosp + E_theta*coso*cosp - E_phi*sinp
E_y = E_r*sino*sinp + E_theta*coso*sinp + E_phi*cosp
E_z = E_r*coso - E_theta*sino

E_abs2 = (np.abs(E_x)**2 + np.abs(E_y**2) + np.abs(E_z)**2)    
E_abs2 = E_abs2/np.nanmax(E_abs2)
    
end = time.perf_counter()

runtime = end - start 

print(runtime)

#%% 
plt.clf()

plt.pcolormesh(X1,Y1,E_abs2, cmap ="jet")        
plt.gca().set_aspect("equal")
plt.colorbar()
plt.show()

#%%

plt.clf()
plt.pcolormesh(X2,Z2,E_abs2, cmap ="jet")        
plt.gca().set_aspect("equal")
plt.colorbar()
plt.show()

#%%

# plt.clf()

# test = bsc_bessel(k0, rho1, phi1, z1, 11, 7, l0,pol0, NA0)[0] 
# plt.pcolormesh(X1,Y1,np.abs(test), cmap = "jet")
# plt.gca().set_aspect("equal")
# plt.colorbar()



    


