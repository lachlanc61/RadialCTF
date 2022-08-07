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
centrex=167 #beam centre position x 167 y 167 for iso2.tif
centrey=167 #beam centre position 256/256 for carlos images

#167-8x 175-6y

#radial params
centrecut=10     #minimum radius (centre mask)
radcut=150      #maximum radius
secwidth=180     #width of sector
secmid=0        #centre of first sector
secstep=180      #step between sectors

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

#rotate image
#not used currently - alternative to rotating mask
def rotate_image(image, angle):
  image_center = tuple(np.array(image.shape[1::-1]) / 2)
  rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
  result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
  return result


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



def ctfmodel(x, amp, Cs, wl, dz, dec, c):
    y = np.zeros_like(x)
    y=-amp*(1/(dec*k))*np.sin( (np.pi/2)*(Cs*wl**3*x**4 - 2*dz*wl*x**2))+c
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
        rpoints=radcut-centrecut-1

        imgmaster = cv2.imread(f, 0)
        profiles=np.zeros((len(steps),rpoints,2))
        
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



        #get the centre and centremask
            center, ccut = (centrex, centrey), centrecut
            
            #   create the azimuthal profile (x,rad) and add to master matrix
            rad = radial_profile(img, center)
            x = np.arange(rad.shape[0])
            rad=rad[ccut:(radcut)]
            x=x[ccut:(radcut)]
            rad=rad[:rpoints]
            x=x[:rpoints]
            profiles[j,:,:] = np.c_[x, rad]
            
        #   PLOTS
            #plot data, found peaks, fits
            
            #guesses - initial guesses for params
            dist=167 #mm? nm?
            amp=10000
            Cs=2    #spherical aberration coeff, mm
            wl=0.001 #wavelength (accel voltage)
             #de broglie eqn eg. https://www.ou.edu/research/electron/bmz5364/calc-kv.html
            dz=1   #defocus value (depth of field)
            dec=1
            c=100

            wl2=0.025
            th=np.arctan(x/dist)
            k=2*np.pi*np.sin(th)/wl2    #spatial frequency in nm ?



     #       https://stackoverflow.com/questions/22895794/scipys-optimize-curve-fit-limits
     #      bounded doesn't work - think only works for two params
     #      try lmfit in so link
            guess=np.array([amp, Cs, wl, dz, dec, c])
            bounded=([amp/10,amp,amp*10],[Cs/10,Cs,Cs*10],[wl/10,wl,wl*10],[dz/10,dz,dz*10],[dec/10,dec,dec*10],[c/10,c,c*10])
            popt, pcov = curve_fit(ctfmodel, k, rad, p0=guess, ftol=0.00001)
            popt, pcov = curve_fit(ctfmodel, k, rad, bounds=bounded)
            print("guess",*guess)
            print("opt",*popt)
            
            print(k[:30])
            ctf=ctfmodel(k, *popt)
           # ctf=ctfmodel(k, *guess)

            # plot the image
            axrad=fig.add_axes([0.08+0.217*j,0.52,0.20,0.45])
            axrad.spines[:].set_linewidth(2)
            axrad.spines[:].set_color(colorVal)
            axrad.set_xticklabels([])
            axrad.set_yticklabels([])
            axrad.tick_params(color=colorVal, labelcolor=colorVal)
            axrad.imshow(img)


            axgph.plot(k, rad, label=secmid, color=colorVal)
            axgph.plot(k, ctf, 
                    ':',
                    color="green")
            """"
            CTF start here
            ctf= np.arange(x.shape[0])

            ctf=-2*sin( (np.pi/2)*(Cs*wl**3*k**4 -2*dz*wl*k**2))

            ctf=-2*sin( (np.pi/2)*(Cs*wl**3*k**4 -2*dz*wl*k**2))
            """
    #FINAL PLOT
    #adjust labels, legends etc

            axgph.set_ylabel('Intensity')
            axgph.set_xlabel('k (nm)')
#            axgph.set_xlim(0,5)
            axgph.legend(loc="upper right")
            j=j+1
        #end for i    
            

        #calculate stats for each image from profiles
        #eg. sum, average, std dev, variance
        """
        psum=np.zeros(rpoints)
        for i in np.arange(len(steps)):
            psum=np.add(psum,profiles[i,:,1])

        pavg=psum/len(steps)

        sq=np.zeros([len(steps),rpoints])
        sqd=np.zeros([len(steps),rpoints])

        
        for i in np.arange(len(steps)):
            sq[i,:]=np.subtract(profiles[i,:,1],pavg)
            sqd[i,:]=np.multiply(sq[i,:],sq[i,:])

        var=np.zeros(rpoints)
        for i in np.arange(len(steps)):
            var=np.add(var,sqd[i,:])

        sd=np.sqrt(var)

        #variance here        
        varval=np.sum(var)/len(var)
        
    #   Add variance to master plot y2

        #initialise variance line
        colorVal = scalarMap.to_rgba(j)
        vline=np.zeros(rpoints)
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
        """
    #   output the final figure for this file
        fig.savefig(os.path.join(odir, ("out_%s.png" % fname)), dpi=300)
        plt.show()
    #   add stats to output matrices
        fnames[h]=fname
#        vars[h]=Decimal(round(varval,2))

    #   clear the figure
        #fig.clf()

        h=h+1

#print the final list of values and save to report file
print(fnames)
print(vars)


np.savetxt(os.path.join(odir, "results.txt"), np.c_[fnames, vars], newline='\n', fmt=['%12s','%12s'], header="      file     variance")

#---------------------------------
#END
#---------------------------------