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

amp=50
dist=20 #mm? nm?
Cs=2    #spherical aberration coeff, mm
wl=0.0025 #wavelength (accel voltage)
        #de broglie eqn eg. https://www.ou.edu/research/electron/bmz5364/calc-kv.html
dz=1000   #defocus value (depth of field)

x=np.arange(0,256,1)
th=np.arctan(x/dist)
k=th    #spatial frequency in nm ?

print(k)



#k=np.arange(0,2,0.01)
y=np.sin(x)

ctf=-amp*np.sin( (np.pi/2)*(Cs*wl**3*k**4 -2*dz*wl*k**2))

plt.plot(k, ctf)

plt.show()

exit()


ctf=-2*sin( (np.pi/2)*(Cs*wl**3*k**4 -2*dz*wl*k**2))