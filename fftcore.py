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
#global variables
FOURIERSPACE=True     #is input in fourierspace already?
FTYPE=".tif"    #valid: ".avi" or ".tif"
DEBUG=False     #debug flag 

#workdir and inputfile
wdirname='data'     #working directory relative to script
odirname='out'      #output directory relative to script
infile = "as_amC_stable.avi"    #assign input file
                                #   only used if reading .avi

#properties of input file
pxpitch=1.252       #nm per pixel in real space
pxdim=512           #width of RS image (should really calc this directly...)
etime=0.1           #seconds per frame

#figure params
colourmap='Set1'    #colourmap for figure
#colourmap='copper'    #colourmap for figure
figx=20         #cm width of figure
figy=10         #cm height of figure
smallfont = 8   #default small font
medfont = 10    #default medium font
lgfont = 12     #default large font
lwidth = 1  #default linewidth
bwidth = 1  #default border width

#radial params
centrecut=5     #minimum radius (centre mask)
secwidth=90     #width of sector
secstep=45      #step between sectors

#initial guesses for fit function
amp=20          #amplitude
Cs=2.7E6        #spherical aberration coeff, nm (=2.7 mm)
wl=0.00335      #wavelength (nm) 
    #from accel voltage via de broglie eqn eg. https://www.ou.edu/research/electron/bmz5364/calc-kv.html
dz=27500        #defocus value (depth of field)  #FIT THIS
dm=130          #damping param
dec=20          #decay param
c=30            #constant
gsig=0.035  #gaussian sigma
gamp=40     #gaussian amplitude

#fit bounding params
bf3=99      #very free
bf2=2       #constrained     
bf1=1.3     #highly constrained
bf0=1.01    #effectively fixed

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

#-----------------------------------
#INITIALISE
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
plt.rc('axes', labelsize=medfont)       # fontsize of the x and y labels
plt.rc('xtick', labelsize=smallfont)    # fontsize of the tick labels
plt.rc('ytick', labelsize=smallfont)    # fontsize of the tick labels
plt.rc('legend', fontsize=smallfont)    # legend fontsize
plt.rc('figure', titlesize=lgfont)      # fontsize of the figure title
plt.rc('lines', linewidth=lwidth)
plt.rcParams['axes.linewidth'] = bwidth

#-----------------------------------
#MAIN START
#-----------------------------------

#read in either .avi or .tif files
#   paired with if/else at beginning of frame-by-frame read
#   clunky but best I can think of so far
#       some possibility to have orphaned variables - eg. vidcap doesn't exist if filetype is tif

#if filetype is avi, read frame-by-frame from avi
if FTYPE == ".avi":
    f = os.path.join(wdir,infile)
    fname = os.path.splitext(os.path.basename(f))[0]

    #assign video capture
    vidcap = cv2.VideoCapture(f)

    #get total number of frames in avi 
    #https://stackoverflow.com/questions/25359288/how-to-know-total-number-of-frame-in-a-file-with-cv2-in-python
    #   - maybe different in more recent version of cv2
    nframes = int(cv2.VideoCapture.get(vidcap, int(cv2.CAP_PROP_FRAME_COUNT)))
    print("opening .avi:",fname)

#if filetype is tif, read as stack of tifs
elif FTYPE == ".tif":
    #read list of files and length of list
    #   glob is random order so need to sort as well
    flist=sorted(glob.glob(wdir + "/*.tif"))
    nframes=len(flist)

    #assign first file and get name 
    #   purely for consistency w/ avi process, likely don't need
    f=flist[0]
    fname = os.path.splitext(os.path.basename(f))[0]

    print("2",nframes)
    print("opening .tifs beginning with:", f)

#if filetype is not avi or tif, throw error
else: 
    print(f'FATAL: filetype {FTYPE} not recognised')
    exit()

print("no frames:", nframes)

#initalise result arrays
times= np.empty(nframes, dtype="U10")    #timestamps
zavg=np.zeros(nframes)                   #zero point average
zsd=np.zeros(nframes)                    #zero point std dev
secsd=np.zeros(nframes)                  #std dev between sectors
r2avg=np.zeros(nframes)                  #average r2 value for fit

#initialise tracking vars
framecount = 0
success = True

#--------------------
#READ frame by frame
#   while prev frame successfully read
#--------------------
while success:

    #initialise plot and colourmaps per frame
    plt.rcParams["figure.figsize"] = [figx/2.54, figy/2.54]
    plt.rcParams["figure.figsize"] = [figx/2.54, figy/2.54]
    fig=plt.figure()
    steps=np.arange(0, 180, secstep)    #no. steps for radial masks
    lut = cm = plt.get_cmap(colourmap) 
    cNorm  = colors.Normalize(vmin=0, vmax=len(steps)+2)
    scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=lut)

    #initialise frame plot
    axgph=fig.add_axes([0.08,0.1,0.85,0.4])

    #READ NEXT FRAME
    #filetype switcher again - read .avi and .tif differently
    #   paired with switcher at top of main 
    #   - be very careful that all branches send the same results downstream
    if FTYPE == ".avi":
        #read a frame from the avi, returning success
        success,readimage = vidcap.read()
    elif FTYPE == ".tif":    
        #read image, provided current frame is less than total frames
        if framecount < len(flist):
            #assign working .tif file
            f=flist[framecount]

            #read in the image
            readimage = cv2.imread(f, 0)

            #return success state based on whether readimage has data
            if readimage.size >= 0:
                success=True
            else:
                print("failed for",f)
                success=False
        #if current frame higher than filelist, report failure
        else:
            success=False
    else:
        print("FATAL: filetype {%} not recognised",FTYPE)
        exit()    

    print('Read frame : ', framecount, success)

    #leave loop if import unsuccessful (ie. no more frames)
    if not success:
        break

    # Check if image is rgb (shape=3 means 3-channels)
    #   and convert to grayscale if needed
    # error if more than 3 channels (eg. alpha channel)
    # continue as-is if two channels

    if len(readimage.shape) == 3:
        #readimage = readimage[:, :, :3].mean(axis=2) 	#old conversion, not sure on difference
        readimage = cv2.cvtColor(readimage, cv2.COLOR_BGR2GRAY)

    elif len(readimage.shape) > 3 :
        print("FATAL: unrecognised format, too many channels in frame:", len(readimage.shape))
         
    #if input is not in fourierspace
    #   do an FFT
    #   otherwise assign ftimage as readimage
    if FOURIERSPACE == False:
        # Array dimensions (array is square) and centre pixel
        # Use smallest dimension and ensure value is odd
        array_size = min(readimage.shape) - 1 + min(readimage.shape) % 2

        # Crop image to square
        readimage = readimage[:array_size, :array_size]
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

        # fourier transform this frame
        ftimage = (abs(calculate_2dft(readimage)))
    else:   #if we don't need an FT
        #assign input as output FT
        ftimage=readimage
    
    if DEBUG == True:
    # Show grayscale image and its Fourier transform
        plt.set_cmap("gray")
        plt.subplot(121)
        plt.imshow(readimage)
        plt.axis("off")
        plt.subplot(122)
    #    print(ftimage)
    #    print(np.abs(ftimage))
    #    cv2.imwrite(os.path.join(odir, "fft_%s.tif" % fname),np.abs(ft))
        plt.imshow(np.log(abs(ftimage)))
        plt.axis("off")
        plt.pause(2)
    
        plt.show()
    
    
# get image centres, outer radius
    centrex, centrey = tuple(np.array(ftimage.shape[1::-1]) / 2)
    centrex=int(centrex)
    centrey=int(centrey)
    radcut=int(max(centrex, centrey)*0.9)
    rpoints=radcut-centrecut-1
    
    #initialise the profile array
    profiles=np.zeros((len(steps),rpoints,2))
    ctfs=np.zeros((len(steps),rpoints))
    zvals=np.zeros(len(steps))
    r2=np.zeros(len(steps))

    stepcount=0 #stepcounter
#   create series of radial masks
#       iterate through each mask position according to secwith and secstep
    for secpos in steps:

        #duplicate the image
        img = np.copy(ftimage)
        colorVal = scalarMap.to_rgba(stepcount)
        #initialise mask from sector coords
        th1=secpos-secwidth/2
        th2=secpos+secwidth/2
        #generate the paired masks
        mask = sector_mask(img.shape,(centrex,centrey),radcut,(th1,th2))
        mask += sector_mask(img.shape,(centrex,centrey),radcut,(th1+180,th2+180))
        #add masks on centre xy column/rows
        mask[:,centrey] = False
        mask[centrex,:] = False
        #apply mask               
        img[~mask] = 0

    #   get the centre and centremask
        center, ccut = (centrex, centrey), centrecut
        
    #   create the azimuthal profile (x,rad) and add to master matrix for this image
        rad = radial_profile(img, center)
        x = np.arange(rad.shape[0])
        rad=rad[ccut:(radcut)]
        x=x[ccut:(radcut)]
        rad=rad[:rpoints]
        x=x[:rpoints]
        profiles[stepcount,:,:] = np.c_[x, rad]

    #   https://stackoverflow.com/questions/22895794/scipys-optimize-curve-fit-limits
    #   FITTING HERE
    #--------------------------------------------------------
        # k=x
        k=x/(pxpitch*pxdim)  
        
        guess=np.array([amp, Cs, wl, dz, dm, dec, c, gsig, gamp])

        bounded=([amp/bf3, Cs/(bf0), wl/bf0, dz/bf2, dm/bf2, dec/bf3, c/bf3, gsig/bf2, gamp/bf2], [amp*bf2, Cs*bf0, wl*bf0, dz*bf2, dm*bf2, dec*bf2, c*bf1, gsig*bf2, gamp*bf2])

    #   DO FIT 
        popt, pcov = curve_fit(ctfmodel, k, rad, p0=guess, bounds=bounded)
    #--------------------------------------------------------

        #populate final models
        ctf=ctfmodel(k, *popt)
        ctfs[stepcount,:]=ctf

        #create model for sin component
        sinfac=np.sin( (np.pi/2)*(popt[1]*popt[2]**3*k**4 - 2*popt[3]*popt[2]*k**2))

        #get index of first crossing of x-axis -> this is the zero point
        zpoint=zerocross(sinfac)[0]
        zvals[stepcount]=k[zpoint]

    #   calc r2 value as rough goodness-of-fit

        # residual sum of squares
        ss_res = np.sum((rad - ctf) ** 2)

        # total sum of squares
        ss_tot = np.sum((rad - np.mean(rad)) ** 2)

        # r-squared
        r2[stepcount] = 1 - (ss_res / ss_tot)

        #   PLOTS
        #plot data, fits, zeropoint

        axrad=fig.add_axes([0.08+0.217*stepcount,0.52,0.20,0.45])
        axrad.spines[:].set_linewidth(2)
        axrad.spines[:].set_color(colorVal)
        axrad.set_xticklabels([])
        axrad.set_yticklabels([])
        axrad.tick_params(color=colorVal, labelcolor=colorVal)
        axrad.imshow(img)
        axgph.plot(k, rad,
            label="%d deg" % secpos,
            color=colorVal)
        axgph.plot(k, ctf, 
            ':',
            color=colorVal)

        axgph.axvline(x=k[zpoint], color=colorVal, linestyle='--')    
        axgph.text(k[zpoint]*1.05,0.95*max(ctf),
            ("%.3f $nm^{-1}$" % k[zpoint]),
            horizontalalignment='left',
            color=colorVal)
        stepcount += 1   #increment stepcounter

    #FINAL PLOT per tiff

    #adjust labels, legends etc    
    axgph.set_ylabel('Intensity')
    axgph.set_xlabel('Spatial frequency (1/nm)')
    axgph.legend(loc="upper right")
   
    #add stats to output matrices
    times[framecount]=etime*framecount
    zavg[framecount]=np.average(zvals)
    zsd[framecount]=np.std(zvals)
    r2avg[framecount]=np.average(r2)

#   would be useful to report difference between sector ctfs as well
#       could re-use variance code removed last commit
    #plt.show()
 #output the final figure for this frame
    fig.savefig(os.path.join(odir, ("out_%s.png" % framecount)), dpi=300)

#   clear and close the figure
#       note: this is slow, costs about 20% performance
#       ideally clear individual axes instead eg. axgh.cla() axrad.cla()
#       some kind of bug with this right now
    fig.clf()
    plt.close()

    framecount += 1

#    if framecount > 5:
#        break
    
#print the final list of report values and save to  file
#   need to work on formatting here, can't print as eg. float because np array has to be single dtype
np.savetxt(os.path.join(odir, "results.txt"), np.c_[times, r2avg, zavg, zsd], newline='\n', fmt=['%12s','%12s','%12s','%12s'], header="      time       r2               zero avg          zero var")

print("CLEAN END")