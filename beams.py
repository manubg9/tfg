import multipolos as mp
import numpy as np
import scipy.integrate as integ
import scipy.special as sp
import matplotlib.pyplot as plt
from math import factorial 
"""
Tightly focused laguerre beams

"""

# def cyl_sph(rho, phic, z):
    
# obtengo eta y gamma para luego calcular la constante que precede a las 
# integrales
def eta_f(k, f):
    
    return -1j*k*f*np.exp(1j*k*f)/(2*np.pi)

def gamma_nm(n, m):
    
    a = 1/(4*np.pi)
    b = (2*n+1)/(n*(n+1))
    c = (factorial(n+m))/(factorial(n-m))
    return np.sqrt(a*b*c)
 
def Ppl(x, p, l):
    
    a = np.sqrt(2)*x
    b = sp.assoc_laguerre(2*x**2, p, np.abs(l))
    c = np.exp(-x**2)
    
    return (a**(np.abs(l)))*b*c
# Aquí separo las integrales para aenm y anmn en dos funciones diferentes
# porque no estoy seguro si quad_vec puede trabajar con funciones que tienen 
# múltiples salidas

def int_a_nm_tflg1(alpha, k, rho, phi, z, n, m, l, p, f, w, pol):
    
    coeff = -4*(np.pi**2)*eta_f(k,f)*gamma_nm(n,m)*((1j)**(l+n-m+1))*np.exp(1j*(l-m)*phi)
    
    arg = f/w * np.sin(alpha)
    comm = np.sin(alpha)*Ppl(arg,p,l)* np.exp(1j*z*np.cos(alpha))
    px, py = pol[0], pol[1]
    sigma = k*rho*np.sin(alpha)
    coeff1 = (px+1j*py)*np.exp((-1j)*phi)*sp.jn(l-m-1, sigma)
    coeff2 = (px-1j*py)*np.exp(1j*phi)*sp.jn(l-m+1, sigma)  
    a1 = mp.pi_nm(alpha,n,m) - mp.tau_nm(alpha, n, m)
    # b1 = -a1
    a2 = mp.pi_nm(alpha,n,m) + mp.tau_nm(alpha, n, m)
    # b2 = a2
    aenm = comm*coeff*(coeff1*a1 + coeff2*a2)
    # amnm = comm*(coeff1*b1 + coeff2*b2)

    return aenm

def int_a_nm_tflg2(alpha, k, rho, phi, z, n, m, l, p, f, w, pol):
    
    coeff = -4*(np.pi**2)*eta_f(k,f)*gamma_nm(n,m)*(1j**(l+n-m+1))*np.exp(1j*(l-m)*phi)

    # Obtengo la parte común del integrando
    arg = f/w * np.sin(alpha)
    comm = np.sin(alpha)*Ppl(arg,p,l)* np.exp(1j*z*np.cos(alpha))
    px, py = pol[0], pol[1]  # guardo los estados de polarización
    sigma = k*rho*np.sin(alpha)  # guardo sigma
    # calculo lo que precede a los "vectores" de comb lineal de tau y pi
    coeff1 = (px+1j*py)*np.exp(-1j*phi)*sp.jn(l-m-1, sigma)
    coeff2 = (px-1j*py)*np.exp(1j*phi)*sp.jn(l-m+1, sigma)  
    # calculo ahora esos vectores para conseguir anmn
    a1 = mp.pi_nm(alpha,n,m) - mp.tau_nm(alpha, n, m)
    b1 = -a1
    a2 = mp.pi_nm(alpha,n,m) + mp.tau_nm(alpha, n, m)
    b2 = a2
    # aenm = comm*(coeff1*a1 + coeff2*a2)
    amnm = comm*coeff*(coeff1*b1 + coeff2*b2)
    
    return amnm

def a_nm_tflg(k, rho, phi, z, n, m, l, p, f, w, pol, a0):
    
    # coeff = -4*(np.pi**2)*eta_f(k,f)*gamma_nm(n,m)*(1j**(l+n-m+1))*np.exp(1j*(l-m)*phi)
    # construyo una tupla con los argumentos restantes del integrando para limpiar un poco el código
    args0 = (k, rho, phi, z, n, m, l, p, f, w, pol)
    # calculo las dos integrales multiplicando por el coeficiente que les precede en cada punto
    aenm  =  integ.quad_vec(int_a_nm_tflg1, 0, a0, args = args0)[0]
    amnm  =  integ.quad_vec(int_a_nm_tflg2, 0, a0, args = args0)[0]
    
    return aenm, amnm
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
# malla de 50x50 en z = 0 con magnitud de lambd0
x1, y1 ,z1 = np.linspace(-3*lb0,3*lb0,100), np.linspace(-3*lb0,3*lb0,100), 0
    
X1, Y1= np.meshgrid(x1,y1)

# calculo cilindricas y esféricas para usar en los BSCs y los multipolos 
rho1 = np.sqrt(X1**2 + Y1**2)
phi1 = np.arctan2(Y1,X1)
phi1 = np.where(phi1 < 0, phi1 + 2*np.pi, phi1)
r1 = np.sqrt(rho1**2 + z1**2)
theta1 = np.arccos(z1/r1)
#%%
# inicio los valores de E de forma que es más sencillo calcular el sumatorio
E_r = 0
E_theta = 0
E_phi = 0


for i in range(1,2):
    for j in range(-i,i+1):

        
        # guardo las componentes en esféricas de de M y N
        A, B, C = mp.M_nm(k0, r1, theta1, phi1, i, j)
        D, F, G = mp.N_nm(k0, r1, theta1, phi1, i, j)
        # Calculo aenm y anmn
        J, K = a_nm_tflg(k0, rho1, phi1, z1, i, j, l0, p0, f0, w0, pol0, np.radians(30))
        # hago el sumatorio independiente de cada componente 
        E_r += J*D + K*A
        E_theta += J*F + K*B
        E_phi += J*G + K*C

        
E_abs2 = (np.abs(E_r)**2 + np.abs(E_theta**2) + np.abs(E_phi)**2)    

#%% 
plt.clf()

plt.pcolormesh(X1,Y1,E_abs2)        
plt.gca().set_aspect("equal")
plt.colorbar()

plt.savefig("test_1.png")




    


