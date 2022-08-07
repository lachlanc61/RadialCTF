from tkinter import X
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cmx
from decimal import *
import cv2
import os
import glob
import scipy.signal as scs

from pathlib import Path
from scipy.optimize import curve_fit



def ctfmodel(x, amp, Cs, wl, dz, dm, dec, c):
    y = np.zeros_like(x)
    y=-amp*(1/(x**dm))*(np.sin( (np.pi/2)*(Cs*wl**3*x**4 - 2*dz*wl*x**2)))-dec*x+c
    return y

def actfmodel(x, amp, Cs, wl, dz, dm, dec, c):
    y = np.zeros_like(x)
    y=abs(-amp*(1/(x**dm))*(np.sin( (np.pi/2)*(Cs*wl**3*x**4 - 2*dz*wl*x**2))))-dec*x+c
    return y


#figure params
colourmap='Set1'    #colourmap for figure
#colourmap='copper'    #colourmap for figure
figx=20         #cm width of figure
figy=10         #cm height of figure
smallfont = 8
medfont = 10
lgfont = 12
lwidth = 1  #default linewidth
bwidth = 1  #default border width


#plot defaults
plt.rc('font', size=smallfont)          # controls default text sizes
plt.rc('axes', titlesize=smallfont)     # fontsize of the axes title
plt.rc('axes', labelsize=medfont)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=smallfont)    # fontsize of the tick labels
plt.rc('ytick', labelsize=smallfont)    # fontsize of the tick labels
plt.rc('legend', fontsize=smallfont)    # legend fontsize
plt.rc('figure', titlesize=lgfont)  # fontsize of the figure title
plt.rc('lines', linewidth=lwidth)
plt.rcParams['axes.linewidth'] = bwidth

plt.rcParams["figure.figsize"] = [figx/2.54, figy/2.54]
plt.rcParams["figure.figsize"] = [figx/2.54, figy/2.54]
fig=plt.figure()

amp=10000
Cs=2    #spherical aberration coeff, mm
wl=0.001 #wavelength (accel voltage)
             #de broglie eqn eg. https://www.ou.edu/research/electron/bmz5364/calc-kv.html
dz=0.5   #defocus value (depth of field)
dm=1 #damping factor
dec=1  #decay factor
c=50     #constant

guess=np.array([amp, Cs, wl, dz, dm, dec, c])

k=np.linspace(1,150,500)
#th=np.arctan(x/dist)

ctf=ctfmodel(k, *guess)
actf=actfmodel(k, *guess)

#plt.plot(k, ctf)
plt.plot(k, actf)

plt.show()

exit()


ctf=-2*sin( (np.pi/2)*(Cs*wl**3*k**4 -2*dz*wl*k**2))