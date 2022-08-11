import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cmx
from decimal import *
import cv2
import os
import glob
import scipy.signal as scs
from scipy.stats import norm

from pathlib import Path
from scipy.optimize import curve_fit



#workdir
wdirname='data'     #working directory relative to script
odirname='out'      #output directory relative to script
infile = "as_amC_stable.avi"

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

#initialise directories relative to script
script = os.path.realpath(__file__) #_file = current script
spath=os.path.dirname(script) 
wdir=os.path.join(spath,wdirname)
odir=os.path.join(spath,odirname)
print("script:", script)
print("script path:", spath)
print("data path:", wdir)

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
#plt.rcParams['figure.dpi'] = 150


#https://stackoverflow.com/questions/33311153/python-extracting-and-saving-video-frames
#https://stackoverflow.com/questions/14275180/read-from-video-avi-file-as-it-is-being-written


f = os.path.join(wdir,infile)

fname = os.path.splitext(os.path.basename(f))[0]
print("opening ",fname)
vidcap = cv2.VideoCapture(f)
success,image = vidcap.read()
framecount = 0
while success:
    #cv2.imwrite(os.path.join(odir, "frame%d.tif" % count),image)     # save frame as TIF file      
    success,image = vidcap.read()
    print('Read frame : ', framecount, success)
    framecount += 1