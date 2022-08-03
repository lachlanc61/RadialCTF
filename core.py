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

#-----------------------------------
#USER MODIFIABLE VARIABLES
#-----------------------------------
#workdir
wdirname='data'     #working directory relative to script
odirname='out'      #output directory relative to script

#fitting
pfprom=10       #prominence threshold for peak fit (default=10)
widthguess=50   #initial guess for peak widths
centrex=185 #beam centre position x 182 185
centrey=183 #beam centre position y 183

#radial params
centrecut=5     #minimum radius (centre mask)
radcut=150      #maximum radius
secwidth=90     #width of sector
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
#-------------------------------------
#FUNCTIONS
#-----------------------------------

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
vars=np.zeros(nfiles)
h=0     #counter

#Interate through files in directory
for ff in os.listdir(wdir):
#for ff in ["r3.tif"]:
    
    
    #split filenames / paths
    f = os.path.join(wdir, ff)
    fname = os.path.splitext(ff)[0]

    #if file is tif, proceed
    if (os.path.isfile(f)) and (f.endswith(".tif")):
        print(fname)
        j=0
        
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
        

    #   read in the image and assign an image centre (manual for now)
        imgmaster = cv2.imread(f, 0)
        profiles=np.zeros((len(steps),radcut-centrecut,2))

        

        if "r0" in fname:
            centrex=185
            centrey=183
        elif "r1" in fname:
            centrex=180
            centrey=182
        elif "r2" in fname:
            centrex=180
            centrey=181    
        elif "r3" in fname:
            centrex=117
            centrey=116
        
    #   iterate through each mask position
        for i in steps:
            print(steps)
            #duplicate the image
            img = np.copy(imgmaster)
            secmid=i
            colorVal = scalarMap.to_rgba(j)
        # initialise mask from sector coords
            th1=secmid-secwidth/2
            th2=secmid+secwidth/2
        # apply the mask
            mask = sector_mask(img.shape,(centrex,centrey),radcut,(th1,th2))
            mask += sector_mask(img.shape,(centrex,centrey),radcut,(th1+180,th2+180))
            img[~mask] = 0

        # plot the image
            axrad=fig.add_axes([0.08+0.217*j,0.52,0.20,0.45])
            axrad.spines[:].set_linewidth(2)
            axrad.spines[:].set_color(colorVal)
            axrad.set_xticklabels([])
            axrad.set_yticklabels([])
            axrad.tick_params(color=colorVal, labelcolor=colorVal)
            axrad.imshow(img)

        #get the centre and centremask
            center, ccut = (centrex, centrey), centrecut
            
            #   create the azimuthal profile (x,rad) and add to master matrix
            rad = radial_profile(img, center)
            rad=rad[ccut:radcut]
            x = np.arange(rad.shape[0])
            profiles[j,:,:] = np.c_[x, rad]
            
        #   PLOTS
            #plot data, found peaks, fits
            
            axgph.plot(x, rad, label=secmid, color=colorVal)
       
            
    #FINAL PLOT
    #adjust labels, legends etc

            axgph.set_ylabel('Intensity')
            axgph.set_xlabel('px (radial)')
            axgph.set_xlim(0,radcut)
            axgph.legend(loc="upper right")
            j=j+1
        #end for i    
        

        #calculate stats for each image from profiles
        #eg. sum, average, std dev, variance

        psum=np.zeros(radcut-centrecut)
        for i in np.arange(len(steps)):
            psum=np.add(psum,profiles[i,:,1])

        pavg=psum/len(steps)

        sq=np.zeros([len(steps),radcut-centrecut])
        sqd=np.zeros([len(steps),radcut-centrecut])

        
        for i in np.arange(len(steps)):
            sq[i,:]=np.subtract(profiles[i,:,1],pavg)
            sqd[i,:]=np.multiply(sq[i,:],sq[i,:])

        var=np.zeros(radcut-centrecut)
        for i in np.arange(len(steps)):
            var=np.add(var,sqd[i,:])

        sd=np.sqrt(var)

        #variance here        
        varval=np.sum(var)/len(var)

    #   Add variance to master plot y2

        #initialise variance line
        colorVal = scalarMap.to_rgba(j)
        vline=np.zeros(radcut-centrecut)
        vline.fill(varval)
        
        axg2=axgph.twinx()
        axg2.plot(x, var, 
                    ':',
                    label=fname, 
                    color="green")
        axg2.plot(x, vline, 
                    '--', 
                    label=fname, 
                    color="green")
        axg2.set_ylabel("Variance",color="green")
        axg2.tick_params(axis="y",colors="green")
        axg2.spines['right'].set_color('green')

        axg2.text(len(vline),2*varval,("variance = %.2f" % varval),horizontalalignment='right',color="green")
        
    #   output the final figure for this file
        fig.savefig(os.path.join(odir, ("out_%s.png" % fname)), dpi=300)

    #   add stats to output matrices
        fnames[h]=fname
        vars[h]=Decimal(round(varval,2))

    #   clear the figure
        fig.clf()

        h=h+1

#print the final list of values and save to report file
print(fnames)
print(vars)

np.savetxt(os.path.join(odir, "results.txt"), np.c_[fnames, vars], newline='\n', fmt=['%12s','%12s'], header="      file     variance")

#---------------------------------
#END
#---------------------------------