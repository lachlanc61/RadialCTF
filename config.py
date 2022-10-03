#-----------------------------------
#USER MODIFIABLE VARIABLES
#-----------------------------------

#global flags
FTYPE=".tif"    #filetype to ingest. Must be ".avi" or ".tif"
FOURIERSPACE=True     #flag whether data is in fourierspace already. will attempt FFT if false
DEBUG=False     #debug flag 

#workdir and inputfile
wdirname='data'     #working directory relative to script
odirname='out'      #output directory relative to script
infile = "alternate/as_amC_stable.avi"    #assign input file
                                #   only used if reading .avi

#properties of input file
pxpitch=1.252       #nm per pixel in real space
pxdim=512           #width of RS image (should really calc this directly...)
etime=0.05           #seconds per frame

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
cs=2.7E6        #spherical aberration coeff, nm (=2.7 mm)
wl=0.00335      #wavelength (nm) 
    #from accel voltage via de broglie eqn eg. https://www.ou.edu/research/electron/bmz5364/calc-kv.html
dz=27500        #defocus value (depth of field)  #FIT THIS
dm=130          #damping param
dec=20          #decay param
const=30            #constant
#const=60           #for 100% radial fit - background is dependent on integration area
gsig=0.035  #gaussian sigma
gamp=40     #gaussian amplitude

#fit bounding params
bf3=99      #very free
bf2=2       #constrained     
bf1=1.3     #highly constrained
bf0=1.01    #effectively fixed
