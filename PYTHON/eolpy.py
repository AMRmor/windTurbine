# -*- coding: utf-8 -*-
"""
Created on Wed May 12 07:14:06 2021

@author: Admin
"""

import numpy as np
from scipy.optimize import fsolve
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

degToRad=np.pi/180

data=np.loadtxt('polnew.dat')
angle=data[:,0]*degToRad
cz=data[:,1]
cx=data[:,2]
cxf=interp1d(angle,cx)
czf=interp1d(angle,cz)

Vz=15;rmin=0.3;rmax=1.8;Lc=0.2;rho=1.2;Npales=3;avd=25*degToRad;avf=25*degToRad
A=15

def va(r,ap):
    return np.sqrt(Vz**2+(r*ap)**2)
def beta(r,ap):
    return np.arcsin(Vz/va(r,ap))
def av(r):
    return avd+(r-rmin)/(rmax-rmin)*(avf-avd)
def i(r,ap):
    return beta(r,ap)-av(r)
#commence ici
#for p in range (134) :
 #   r=((rmax-rmin)/134)*p
    
def sigma(r):
    return Npales*Lc/2*np.pi*r

def A1(ap,r):
    #for p in range (134) :
    # r=((rmax-rmin)/134)*p
     r=np.linspace(rmin,rmax,20)
     anglei=i(r,ap)
     pb=beta(r,ap)
     B12=sigma(r)*(czf(anglei)*np.cos(pb)+cxf(anglei)*np.sin(pb))/4*((np.sin(pb))**2)
     return B12/(B12+1)
    
def A2(r,ap):
    #for p in range (134) :
     #r=((rmax-rmin)/134)*p
     #r=np.linspace(rmin,rmax,20) 
     anglei=i(r,ap)
     pb=beta(r,ap)
     B22=sigma(r)*(czf(anglei)*np.sin(pb)-cxf(anglei)*np.cos(pb))/4*np.sin(pb)*np.cos(pb)
     return B22/(1-B22)


def f(r,ap):
    anglei=i(r,ap)
    pb=beta(r,ap)
    term1=-cxf(anglei)*np.cos(pb)
    term2=czf(anglei)*np.sin(pb)
    term3=0.5*rho*va(r,ap)**2*Lc
    return (term1+term2)*term3

def Caero(ap):
    r=np.linspace(rmin,rmax,20)
    y=r*f(r,ap)
    It=np.trapz(y,r)
    return It

def g(ap):
    return Npales*Caero(ap)-A*np.abs(ap)

plt.figure("g")
ap=np.linspace(1,15)
yg=np.array([g(api) for api in ap])
plt.plot(ap,yg)

vitrot=fsolve(g,2)
print(vitrot)




coefA1=fsolve(A1,0)
coefA2=fsolve(A2,0)
print(coefA1)
print(coefA2)




