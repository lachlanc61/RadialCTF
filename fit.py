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

#-----------------------------------
#USER MODIFIABLE VARIABLES
#-----------------------------------
#workdir
wdirname='data'     #working directory relative to script
odirname='out'      #output directory relative to script

#radial params
centrecut=5     #minimum radius (centre mask)
secwidth=45     #width of sector
secmid=0        #centre of first sector
secstep=45      #step between sectors

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


#initial guesses for fit params
pxpitch=1.252    #nm per pixel in real space
pxdim=512
"""
https://www.dsprelated.com/showthread/comp.dsp/24594-1.php
In each dimension, units are "cycles per aperture width". 

If an image of a scene represents M=4 meters by N=3 meters of area, then any
element (m,n) of the output FFT is m cycles per 4 meters by n cycles per 3
meters.

guess 50.06 2700000.0 0.00335 26500 150 12.95 56
"""

amp=20    #amplitude
Cs=2.7E6    #spherical aberration coeff, nm (=2.7 mm)
wl=0.00335 #wavelength (nm) 
    #from accel voltage via de broglie eqn eg. https://www.ou.edu/research/electron/bmz5364/calc-kv.html
dz=27500   #defocus value (depth of field)  #FIT THIS
dm=130        #damping param
dec=20   #decay param
c=30        #constant

gsig=0.035
gamp=40

#bounding factors - define fit limits
bf3=99
bf2=2        
bf1=1.3
bf0=1.01

#to-do change damping param to true envelope function 
#   http://blake.bcm.tmc.edu/eman1/ctfc/ctfc.html
# e^(-Bk**2)

#true constant/decay is more complicated as well, bx+c probably good enough for now

""""
MANUAL GUESS - defocus val is real fit param - effecitvely changes phase
amp=50    #amplitude
Cs=2.7E6    #spherical aberration coeff, nm (=2.7 mm)
wl=0.00335 #wavelength (nm) 
#wl=0.1 #wavelength (nm) 
    #from accel voltage via de broglie eqn eg. https://www.ou.edu/research/electron/bmz5364/calc-kv.html
dz=26500   #defocus value (depth of field)  #FIT THIS
dm=150        #damping param
dec=13   #decay param
c=56        #constant

gsig=0.03
gamp=80

#bounds
bf2=2        #bounding factor - defines fit limits
bf1=1.3
bf0=1.01

OPT
amp=25    #amplitude
Cs=2.7E6    #spherical aberration coeff, nm (=2.7 mm)
wl=0.00335 #wavelength (nm) 
    #from accel voltage via de broglie eqn eg. https://www.ou.edu/research/electron/bmz5364/calc-kv.html
dz=27500   #defocus value (depth of field)  #FIT THIS
dm=130        #damping param
dec=20   #decay param
c=60        #constant

gsig=0.035
gamp=80

#bounds
bf2=2        #bounding factor - defines fit limits
bf1=1.3
bf0=1.01
"""
#-------------------------------------
#FUNCTIONS
#-----------------------------------

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
#    y=abs(-2*amp*(1/(x**dm))*(np.sin( (np.pi/2)*(Cs*wl**3*x**4 - 2*dz*wl*x**2))))-dec*x+c
#    print(-dm*x**2, np.exp(-dm*x**2))
    y=ngauss(x, 0, gsig, gamp)+2*amp*(np.exp(-dm*x**2))*abs(np.sin( (np.pi/2)*(Cs*wl**3*x**4 - 2*dz*wl*x**2)))-dec*x+c
    return y

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

#sanitise secmid and secwidth
if (secmid < 0) or (secmid >= 360):
    print("FATAL: sector centre = {} deg is OUT OF RANGE. Expected value between 0 and 360 (2*pi)".format(secmid))
    exit()


#initialise file# and report matrices
nfiles=len(glob.glob1(wdir,"*.tif"))
fnames= np.empty(nfiles, dtype="U10")
zavg=np.zeros(nfiles)    #zero point average
zvar=np.zeros(nfiles)   #zero point variance
r2avg=np.zeros(nfiles)  #average r2 

h=0 #initialise filecounter
#Interate through files in directory
for ff in os.listdir(wdir):
#for ff in ["r3.tif"]:
    
    
    #split filenames / paths
    f = os.path.join(wdir, ff)
    fname = os.path.splitext(ff)[0]

    #if file is tif, proceed
    if (os.path.isfile(f)) and (f.endswith(".tif")):
        print(fname)
        

        #initialise plot and colourmaps per file
        steps=np.arange(0, 180, secstep)
        plt.rcParams["figure.figsize"] = [figx/2.54, figy/2.54]
        plt.rcParams["figure.figsize"] = [figx/2.54, figy/2.54]
        fig=plt.figure()

        lut = cm = plt.get_cmap(colourmap) 
        cNorm  = colors.Normalize(vmin=0, vmax=len(steps)+2)
        scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=lut)

        #initialise primary plot
        axgph=fig.add_axes([0.08,0.1,0.85,0.4])

        #initialise image printouts
        
    #   read in the image
        imgmaster = cv2.imread(f, 0)
        # get image centres, outer radius
        centrex, centrey = tuple(np.array(imgmaster.shape[1::-1]) / 2)
        centrex=int(centrex)
        centrey=int(centrey)
        radcut=int(max(centrex, centrey)*0.9)
        rpoints=radcut-centrecut-1
        
        #initialise the profile array
        profiles=np.zeros((len(steps),rpoints,2))
        ctfs=np.zeros((len(steps),rpoints))
        zvals=np.zeros(len(steps))
        r2=np.zeros(len(steps))

        j=0 #stepcounter
    #   create series of radial masks
    #       iterate through each mask position according to secwith and secstep
        for i in steps:
        #    print(steps)
        #   duplicate the image
            img = np.copy(imgmaster)
            secmid=i
            colorVal = scalarMap.to_rgba(j)
        # initialise mask from sector coords
            th1=secmid-secwidth/2
            th2=secmid+secwidth/2
        # generate the paired masks
            mask = sector_mask(img.shape,(centrex,centrey),radcut,(th1,th2))
            mask += sector_mask(img.shape,(centrex,centrey),radcut,(th1+180,th2+180))
        # add masks on centre xy column/rows
            mask[:,centrey] = False
            mask[centrex,:] = False
        # apply mask               
            img[~mask] = 0



        #get the centre and centremask
            center, ccut = (centrex, centrey), centrecut
            
            #   create the azimuthal profile (x,rad) and add to master matrix for this image
            rad = radial_profile(img, center)
            x = np.arange(rad.shape[0])
            rad=rad[ccut:(radcut)]
            x=x[ccut:(radcut)]
            rad=rad[:rpoints]
            x=x[:rpoints]
            profiles[j,:,:] = np.c_[x, rad]



     #       https://stackoverflow.com/questions/22895794/scipys-optimize-curve-fit-limits
     #      FITTING HERE
     #--------------------------------------------------------
           # k=x
            k=x/(pxpitch*pxdim)  
            
            guess=np.array([amp, Cs, wl, dz, dm, dec, c, gsig, gamp])
            #insin=(guess[1]*guess[2]**3*k**4 - 2*guess[3]*guess[2]*k**2)
            #sinfac=np.sin( (np.pi/2)*insin)
            #ampfac=2*amp*(np.exp(-dm*k**2))
            #oampfac=2*amp*(1/(x**2))
            #baseline=-dec*k+c
   
            bounded=([amp/bf3, Cs/(bf0), wl/bf0, dz/bf2, dm/bf2, dec/bf3, c/bf3, gsig/bf2, gamp/bf2], [amp*bf2, Cs*bf0, wl*bf0, dz*bf2, dm*bf2, dec*bf2, c*bf1, gsig*bf2, gamp*bf2])
            #print("guess", amp, Cs, wl, dz, dm, dec, c)

            # DO FIT 
            popt, pcov = curve_fit(ctfmodel, k, rad, p0=guess, bounds=bounded)
     #--------------------------------------------------------

            #create final model
            ctf=ctfmodel(k, *popt)
            ctfs[j,:]=ctf


            
#           mdiff=np.subtract(rad, ctf)
#           print(np.average(mdiff))
            
            #ctf=ctfmodel(k, *guess)

            #create model for sin component
            sinfac=np.sin( (np.pi/2)*(popt[1]*popt[2]**3*k**4 - 2*popt[3]*popt[2]*k**2))

            #get index of first crossing of x-axis -> this is the zero point
            zpoint=zerocross(sinfac)[0]
            zvals[j]=k[zpoint]

            #calc r2 value as rough goodness-of-fit
             
            # residual sum of squares
            ss_res = np.sum((rad - ctf) ** 2)

            # total sum of squares
            ss_tot = np.sum((rad - np.mean(rad)) ** 2)

            # r-squared
            r2[j] = 1 - (ss_res / ss_tot)


        #    print("params: amp, Cs, wl, dz, dm, dec, c")
        #    print("guess",*guess)
        #    print("opt",*popt)
        #    print("zero point",k[zpoint])

            #   PLOTS
            #plot data, fits, zeropoint

            axrad=fig.add_axes([0.08+0.217*j,0.52,0.20,0.45])
            axrad.spines[:].set_linewidth(2)
            axrad.spines[:].set_color(colorVal)
            axrad.set_xticklabels([])
            axrad.set_yticklabels([])
            axrad.tick_params(color=colorVal, labelcolor=colorVal)
            axrad.imshow(img)
            axgph.plot(k, rad,
                label="%d deg" % secmid,
                color=colorVal)
            axgph.plot(k, ctf, 
                ':',
                color=colorVal)

            axgph.axvline(x=k[zpoint], color=colorVal, linestyle='--')    
            axgph.text(k[zpoint]*1.05,0.95*max(ctf),
                ("%.3f $nm^{-1}$" % k[zpoint]),
                horizontalalignment='left',
                color=colorVal)
            j=j+1   #increment stepcounter
            #end for i in steps (radial)   

    #FINAL PLOT per tiff
    #adjust labels, legends etc
        
        axgph.set_ylabel('Intensity')
        axgph.set_xlabel('Spatial frequency (1/nm)')
        axgph.legend(loc="upper right")
    
    #   output the final figure for this file
        fig.savefig(os.path.join(odir, ("out_%s.png" % fname)), dpi=300)
        
    #   add stats to output matrices
        fnames[h]=fname
        zavg[h]=np.average(zvals)
        zvar[h]=np.var(zvals)
        r2avg[h]=np.average(r2)
    #   would be useful to report difference between sector ctfs as well
    #       could re-use variance code removed last commit

    #   clear and close the figure
        
        fig.clf()
        plt.close()

    # this is slow - about 20% performance to close after each print
    #   better way would be to move fig and axes initialisations to above loop,  
    #   then axgh.cla() axrad.cla() to clear axes only
    #   - currently this leaves tickmarks on last radial profile
    #       cant work out how to fix right now, lets just do it the slow way
       
        h=h+1

#print the final list of values and save to report file
np.savetxt(os.path.join(odir, "results.txt"), np.c_[fnames, r2avg, zavg, zvar], newline='\n', fmt=['%12s','%12s','%12s','%12s'], header="      file       r2               zero avg          zero var")

#---------------------------------
#END
#---------------------------------