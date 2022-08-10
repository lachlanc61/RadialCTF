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

""""
FFT from:
https://thepythoncodingbook.com/2021/08/30/2d-fourier-transform-in-python-and-fourier-synthesis-of-images/

avi read from:
#https://stackoverflow.com/questions/33311153/python-extracting-and-saving-video-frames
"""
#-----------------------------------
#USER MODIFIABLE VARIABLES
#-----------------------------------
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


#-------------------------------------
#FUNCTIONS
#-----------------------------------

def calculate_2dft(input):
    ft = np.fft.ifftshift(input)
    ft = np.fft.fft2(ft)
    return np.fft.fftshift(ft)

def calculate_distance_from_centre(coords, centre):
    # Distance from centre is √(x^2 + y^2)
    return np.sqrt(
        (coords[0] - centre) ** 2 + (coords[1] - centre) ** 2
    )

def find_symmetric_coordinates(coords, centre):
    return (centre + (centre - coords[0]),
            centre + (centre - coords[1]))



#-----------------------------------
#MAIN START
#-----------------------------------

#initialise directories relative to script
script = os.path.realpath(__file__) #_file = current script
spath=os.path.dirname(script) 
wdir=os.path.join(spath,wdirname)
odir=os.path.join(spath,odirname)
print("script:", script)
print("script path:", spath)
print("data path:", wdir)

#initialise plot defaults
plt.rc('font', size=smallfont)          # controls default text sizes
plt.rc('axes', titlesize=smallfont)     # fontsize of the axes title
plt.rc('axes', labelsize=medfont)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=smallfont)    # fontsize of the tick labels
plt.rc('ytick', labelsize=smallfont)    # fontsize of the tick labels
plt.rc('legend', fontsize=smallfont)    # legend fontsize
plt.rc('figure', titlesize=lgfont)  # fontsize of the figure title
plt.rc('lines', linewidth=lwidth)
plt.rcParams['axes.linewidth'] = bwidth





f = os.path.join(wdir,infile)

fname = os.path.splitext(os.path.basename(f))[0]
print("opening ",fname)
vidcap = cv2.VideoCapture(f)
success,image = vidcap.read()
framecount = 0
while success:
    #cv2.imwrite(os.path.join(odir, "frame%d.tif" % count),image)     # save frame as TIF file      
    success,rsimage = vidcap.read()
    rsimage = rsimage[:, :, :3].mean(axis=2)  # Convert to grayscale
    # Array dimensions (array is square) and centre pixel
    # Use smallest of the dimensions and ensure it's odd
    array_size = min(rsimage.shape) - 1 + min(rsimage.shape) % 2
    # Crop image so it's a square image
    rsimage = rsimage[:array_size, :array_size]
    centre = int((array_size - 1) / 2)
    # Get all coordinate pairs in the left half of the array,
    # including the column at the centre of the array (which
    # includes the centre pixel)
    coords_left_half = (
        (x, y) for x in range(array_size) for y in range(centre+1)
    )
    # Sort points based on distance from centre
    coords_left_half = sorted(
        coords_left_half,
        key=lambda x: calculate_distance_from_centre(x, centre)
    )

    plt.set_cmap("gray")
    ftimage = calculate_2dft(rsimage)

 # Show grayscale image and its Fourier transform
    plt.subplot(121)
    plt.imshow(rsimage)
    plt.axis("off")
    plt.subplot(122)
    print(ftimage)
    print(np.abs(ftimage))
#    cv2.imwrite(os.path.join(odir, "fft_%s.tif" % fname),np.abs(ft))
    plt.imshow(np.log(abs(ftimage)))
    plt.axis("off")
    plt.pause(2)

    plt.show()

    print('Read frame : ', framecount, success)
    framecount += 1
    exit()