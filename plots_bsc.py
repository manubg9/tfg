#%%
import numpy as np
import matplotlib.pyplot as plt
import beams as bsc
import multipolos as mp
import time
import scipy.special as sp
#%%

def plot_beams(btype, grid_size, nmax, lb0, rhop, phip, zp, l, p, f, w, pol, NA):
        
    k = 2*np.pi/lb0 
    
    extent = np.linspace(-2.5*lb0, 2.5*lb0, grid_size)
    # Domain 
    "z=0"
    x1 = extent
    y1 = extent  
    X1, Y1= np.meshgrid(x1,y1)
    Z1 = np.zeros_like(X1)
    rho1 = np.sqrt(X1**2 + Y1**2)
    rho1 = np.where(rho1==0, 1e-16, rho1)
    phi1 = np.arctan2(Y1,X1)
    phi1 = np.where(phi1 < 0, phi1 + 2*np.pi, phi1)
    r1 = np.sqrt(rho1**2 + Z1**2)
    theta1 = np.arccos(Z1/r1)
    
    "y=0"
    x2 = extent
    y2 = 0
    z2 = extent 
    X2, Z2= np.meshgrid(x2,z2)
    Y2 = np.zeros_like(X2)
    rho2 = np.sqrt(X2**2 + Y2**2)
    phi2= np.arctan2(Y2,X2)
    phi2 = np.where(phi2 < 0, phi2 + 2*np.pi, phi2)
    r2 = np.sqrt(rho2**2 + Z2**2)
    r2 = np.where(r2 == 0, 1e-16, r2)
    theta2 = np.arccos(Z2/r2)
    

    
    n = np.arange(1, nmax+1,1)
    m = np.arange(-nmax, nmax + 1,1)
    M, N = np.meshgrid(m,n)
    aeh = np.zeros_like(N, dtype = float)
    amh = np.zeros_like(N, dtype = float)
    
    
    Erz= np.zeros_like(X1, dtype = complex) 
    Etz = np.zeros_like(X1, dtype = complex)
    Epz = np.zeros_like(X1, dtype = complex)
    
    Ery= np.zeros_like(X2, dtype = complex)
    Ety = np.zeros_like(X2, dtype = complex)
    Epy = np.zeros_like(X2, dtype = complex)
    
    for i in range(nmax):
        for j in range(2*nmax + 1):
            nid = N[i,j]
            mid = M[i,j]
            
            if nid >= np.abs(mid):
                print((nid,mid))
                Mrz, Mtz, Mpz = mp.M_nm(k, r1, theta1, phi1, nid, mid)
                Mry, Mty, Mpy = mp.M_nm(k, r2, theta2, phi2, nid, mid)
                Nrz, Ntz, Npz = mp.N_nm(k, r1, theta1, phi1, nid, mid)
                Nry, Nty, Npy = mp.N_nm(k, r2, theta2, phi2, nid, mid)
                
                if btype == "tflg":
                    
                    ae, am = bsc.bsc_tflg(k, rhop, phip, zp, nid, mid, l, p, f, w, pol, NA)
                 
                elif btype == "bessel":
                    
                    ae, am = bsc.bsc_bessel(k, rhop, phip, zp, nid, mid, l, pol, NA)
                
                elif btype =="cyl":
                    
                    ae, am = bsc.bsc_cyl(k, rhop, phip, zp, nid, mid, l, pol, NA)
                
                absae, absam = np.abs(ae), np.abs(am)
                    
                Erz += ae*Nrz + am*Mrz
                Etz += ae*Ntz + am*Mtz
                Epz += ae*Npz + am*Mpz
                    
                Ery += ae*Nry + am*Mry
                Ety += ae*Nty + am*Mty
                Epy += ae*Npy + am*Mpy
                
                aer, aem = absae, absam
             
            else:
                
                aer, aem = 0.0, 0.0
                
            
            aeh[i,j] = aer
            amh[i,j] = aem
            
    E2z = np.abs(Erz)**2 + np.abs(Etz)**2 + np.abs(Epz)**2
    E2y = np.abs(Ery)**2 + np.abs(Ety)**2 + np.abs(Epy)**2
                    
    E2z = E2z/np.nanmax(E2z)
    E2y = E2y/np.nanmax(E2y)
                
    plt.figure(0)
    plt.clf()
    plt.pcolormesh(X1,Y1,E2z, cmap ="hot_r", shading = "auto")
    plt.xlabel(r"$x$ $[\mu m]$")
    plt.ylabel(r"$y$ $[\mu m]$")
    plt.xticks([])
    plt.yticks([])
    plt.gca().set_aspect("equal")
    ref = plt.Rectangle((0.85e-6, -1.8e-6), 1e-6, 0.1e-6, facecolor="black", edgecolor = "white")
    plt.text(1.3e-6, -1.6e-6, r"$\lambda_0$", fontsize = "large")
    plt.gca().add_patch(ref)
    plt.colorbar()
    plt.show()
                
    plt.figure(1)
    plt.clf()
    plt.pcolormesh(X2,Z2,E2y, cmap ="hot_r", shading = "auto")
    plt.xlabel(r"$x$ $[\mu m]$")
    plt.ylabel(r"$z$ $[\mu m]$")
    plt.xticks([])
    plt.yticks([])
    plt.gca().set_aspect("equal")
    ref = plt.Rectangle((0.85e-6, -1.8e-6), 1e-6, 0.1e-6, facecolor="black", edgecolor = "white")
    plt.text(1.3e-6, -1.6e-6, r"$\lambda_0$", fontsize = "large")
    plt.gca().add_patch(ref)
    plt.colorbar()
    plt.show()

            
            
    nx = N.ravel()
    my = M.ravel()
    az = np.zeros_like(nx)
    dx, dy = 1, 1
    dze = aeh.ravel()
    dzm = amh.ravel()
            
    plt.figure(2)
    plt.clf()
    ax = plt.figure(2).add_subplot(projection="3d")
    ax.bar3d(nx,my,az, dx, dy, dze, alpha = 0.9)
    ax.set_zlim3d(0, np.nanmax(dze))  
    plt.xticks([])
    plt.yticks([])
    plt.xlabel(r"$n$")
    plt.ylabel(r"$m$")
    plt.title(r"$\left| a^e_{nm}\right|$")
    plt.show()
        
    plt.figure(3)
    plt.clf()
    ax = plt.figure(3).add_subplot(projection="3d")
    ax.bar3d(nx,my,az, dx, dy, dzm, alpha = 0.9)
    ax.set_zlim3d(0, np.nanmax(dzm))  
    plt.xticks([])
    plt.yticks([])
    plt.xlabel(r"$n$")
    plt.ylabel(r"$m$")
    plt.title(r"$\left| a^m_{nm}\right|$")
    plt.show()
        
                    
                
                
                
#%%

pol0 = (1,0)
pol1 = (1,1j)/np.sqrt(2)
pol2 = (1,-1j)/np.sqrt(2)
pol3 = (0,1)
pol4 = (1,1)
pol5 = (1,-1)
#%%
plot_beams("bessel", 100, 50, 1e-6, 0,0,0,0,0,1e-3,1e-3,pol0, 70)               
               

#%%




plt.figure(2)
plt.gca().set_yticks([-1,1])

#%%

test = np.arange(-50, 50, 1)
p = sp.jn(1 -test - 1,0)
plt.clf()
plt.plot(test, p, "r." )