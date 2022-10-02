import numpy as np
import cv2
import os
import glob

#from decimal import *
import decimal
from scipy.stats import norm
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

import config

def initialise():
    script = os.path.realpath(__file__) #_file = current script
    spath=os.path.dirname(script) 
    spath=os.path.dirname(spath)
    wdir=os.path.join(spath,config.wdirname)
    odir=os.path.join(spath,config.odirname)
    print(
    "---------------------------\n"
    "PATHS\n"
    "---------------------------\n"
    f"base: {spath}\n"
    f"data: {wdir}\n"
    f"output: {odir}\n"
    "---------------------------"
    )

    if True:
        #   initialise plot defaults
        plt.rc('font', size=config.smallfont)          # controls default text sizes
        plt.rc('axes', titlesize=config.smallfont)     # fontsize of the axes title
        plt.rc('axes', labelsize=config.medfont)       # fontsize of the x and y labels
        plt.rc('xtick', labelsize=config.smallfont)    # fontsize of the tick labels
        plt.rc('ytick', labelsize=config.smallfont)    # fontsize of the tick labels
        plt.rc('legend', fontsize=config.smallfont)    # legend fontsize
        plt.rc('figure', titlesize=config.lgfont)      # fontsize of the figure title
        plt.rc('lines', linewidth=config.lwidth)
        plt.rcParams['axes.linewidth'] = config.bwidth

    ftype=config.FTYPE
    #read in either .avi or .tif files
    #   paired with if/else at beginning of frame-by-frame read
    #   clunky but works for now
    #       some possibility to have orphaned variables - eg. vidcap doesn't exist if filetype is tif

    #if filetype is avi, read frame-by-frame from avi
    if ftype == ".avi":
        f = os.path.join(wdir,config.infile)
        fname = os.path.splitext(os.path.basename(f))[0]

        #assign video capture
        vidcap = cv2.VideoCapture(f)

        #get total number of frames in avi 
        #https://stackoverflow.com/questions/25359288/how-to-know-total-number-of-frame-in-a-file-with-cv2-in-python
        #   - maybe different in more recent version of cv2
        nframes = int(cv2.VideoCapture.get(vidcap, int(cv2.CAP_PROP_FRAME_COUNT)))
        print("opening .avi:",fname)

        flist=f

    #if filetype is tif, read as stack of tifs
    elif ftype == ".tif":
        #read list of files and length of list
        #   glob is random order so need to sort as well
        flist=sorted(glob.glob(wdir + "/*.tif"))
        nframes=len(flist)

        #return success state based on whether readimage has data
        if len(flist) == 0:
            print(f'FATAL: no files found in {wdir}')
            exit()

        #assign first file and get name 
        #   purely for consistency w/ avi process, likely don't need
        f=flist[0]
        fname = os.path.splitext(os.path.basename(f))[0]

        print("2",nframes)
        print("opening .tifs beginning with:", f)

        vidcap=None

    #if filetype is not avi or tif, raise error and end
    else: 
        raise TypeError(f'Filetype {ftype} not recognised')

    print("no frames:", nframes)

    return f, fname, script, spath, wdir, odir, vidcap, flist, nframes, ftype

def getimgparams():
    figx=config.figx
    figy=config.figy
    secstep=config.secstep
    colourmap=config.colourmap
    fourierspace=bool(config.FOURIERSPACE)
    debug=bool(config.DEBUG)
    centrecut=config.centrecut
    secwidth=config.secwidth
    pxpitch=config.pxpitch
    pxdim=config.pxdim
    return figx, figy, secstep, colourmap, fourierspace, debug, centrecut, secwidth, pxpitch, pxdim


def getfitparams():
    amp=config.amp
    cs=config.cs
    wl=config.wl
    dz=config.dz
    dm=config.dm
    dec=config.dec
    const=config.const
    gsig=config.gsig
    gamp=config.gamp
    bf0=config.bf0
    bf1=config.bf1
    bf2=config.bf2
    bf3=config.bf3
    etime=config.etime
    return amp, cs, wl, dz, dm, dec, const, gsig, gamp, bf0, bf1, bf2, bf3, etime   

#   do 2D FFT - takes greyscale image as 2D matrix
def calculate_2dft(input):
    ft = np.fft.ifftshift(input)
    ft = np.fft.fft2(ft)
    return np.fft.fftshift(ft)

#calculate distance from centre for coordinates x,y
def calculate_distance_from_centre(coords, centre):
    # Distance from centre is √(x^2 + y^2)
    return np.sqrt(
        (coords[0] - centre) ** 2 + (coords[1] - centre) ** 2
    )

#find symmetric coordinate pair around centre point, given coord to mirror and coord for centre
def find_symmetric_coordinates(coords, centre):
    return (centre + (centre - coords[0]),
            centre + (centre - coords[1]))

#rotate image
#not used currently - alternative to rotating mask
def rotate_image(image, angle):
  image_center = tuple(np.array(image.shape[1::-1]) / 2)
  rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
  result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
  return result

#get index from array closest to value
#https://stackoverflow.com/questions/2566412/find-nearest-value-in-numpy-array
def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx

#get index where component first crosses axis
#https://stackoverflow.com/questions/3843017/efficiently-detect-sign-changes-in-python
def zerocross(array):
    return np.where(np.diff(np.sign(array)))[0]

#Generate a radial profile
def radial_profile(data, center):
#    print(data[center[0],:])
#    print(np.indices((data.shape)))
    y,x = np.indices((data.shape)) # first determine radii of all pixels
    r = np.sqrt((x-center[0])**2+(y-center[1])**2)
   # print(r)
    ind = np.argsort(r.flat) # get sorted indices
    sr = r.flat[ind] # sorted radii
    sim = data.flat[ind] # image values sorted by radii
    ri = sr.astype(np.int32) # integer part of radii (bin size = 1)
    # determining distance between changes
    deltar = ri[1:] - ri[:-1] # assume all radii represented
    rind = np.where(deltar)[0] # location of changed radius
    nr = rind[1:] - rind[:-1] # number in radius bin
    csim = np.cumsum(sim, dtype=np.float64) # cumulative sum to figure out sums for each radii bin
    tbin = csim[rind[1:]] - csim[rind[:-1]] # sum for image values in radius bins
    radialprofile = tbin/nr # the answer
    return radialprofile

#Create a radial sector mask
#based on https://stackoverflow.com/questions/59432324/how-to-mask-image-with-binary-mask
def sector_mask(shape,centre,radius,angle_range):
    """
    Return a boolean mask for a circular sector. The start/stop angles in  
    `angle_range` should be given in clockwise order.
    """

    x,y = np.ogrid[:shape[0],:shape[1]]
    cx,cy = centre
    tmin,tmax = np.deg2rad(angle_range)

    # convert cartesian --> polar coordinates
    r2 = (x-cx)*(x-cx) + (y-cy)*(y-cy)
    theta = -1*np.arctan2(x-cx,y-cy)
    
    # wrap angles between 0 and 2*pi
    theta %= (2*np.pi)

    # circular mask
    circmask = r2 <= radius*radius

    # angular mask
    if (tmin <= 0) and (tmax >= 0):
        #or 360
        tmin = 2*np.pi+tmin
        anglemask = np.logical_and(theta >= tmin,theta <= 2*np.pi) #or (theta <= (tmax+np.pi))
        anglemask += np.logical_and(theta >= 0,theta <= tmax)
    elif (tmin <= 2*np.pi) and (tmax >= 2*np.pi):
        tmax = tmax-2*np.pi
        anglemask = np.logical_and(theta >= tmin,theta <= 2*np.pi) #or (theta <= (tmax+np.pi))
        anglemask += np.logical_and(theta >= 0,theta <= tmax)
    else:
        anglemask = np.logical_and(theta >= tmin%(2*np.pi),theta <= tmax%(2*np.pi)) #or (theta <= (tmax+np.pi))
    
    return circmask*anglemask

#outputs gaussian with max y = amp
def ngauss(x, mu, sig1, amp):
    g1=norm.pdf(x, mu, sig1)
    g1n=np.divide(g1,max(g1))
    return np.multiply(g1n, amp)


def ctfmodel(x, amp, Cs, wl, dz, dm, dec, c, gsig, gamp):
    y = np.zeros_like(x)
    y=ngauss(x, 0, gsig, gamp)+2*amp*(np.exp(-dm*x**2))*abs(np.sin( (np.pi/2)*(Cs*wl**3*x**4 - 2*dz*wl*x**2)))-dec*x+c
    return y