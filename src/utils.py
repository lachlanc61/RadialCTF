import numpy as np
import cv2
import os
import glob
import yaml
import matplotlib.pyplot as plt

from scipy.stats import norm

def readcfg(filename):
    dir = os.path.realpath(__file__) #_file = current script
    dir=os.path.dirname(dir) 
    dir=os.path.dirname(dir)    #second call to get src/..

    yamlfile=os.path.join(dir,filename)

    with open(yamlfile, "r") as f:
        return yaml.safe_load(f)

def initialise(config):
    script = os.path.realpath(__file__) #_file = current script
    spath=os.path.dirname(script) 
    spath=os.path.dirname(spath)    #second call to get src/..
    wdir=os.path.join(spath,config['wdirname'])
    odir=os.path.join(spath,config['odirname'])
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
        plt.rc('font', size=config['smallfont'])          # controls default text sizes
        plt.rc('axes', titlesize=config['smallfont'])     # fontsize of the axes title
        plt.rc('axes', labelsize=config['medfont'])       # fontsize of the x and y labels
        plt.rc('xtick', labelsize=config['smallfont'])    # fontsize of the tick labels
        plt.rc('ytick', labelsize=config['smallfont'])    # fontsize of the tick labels
        plt.rc('legend', fontsize=config['smallfont'])    # legend fontsize
        plt.rc('figure', titlesize=config['lgfont'])      # fontsize of the figure title
        plt.rc('lines', linewidth=config['lwidth'])
        plt.rcParams['axes.linewidth'] = config['bwidth']

    ftype=config['FTYPE']
    #read in either .avi or .tif files
    #   paired with if/else at beginning of frame-by-frame read
    #   clunky but works for now
    #       some risk of orphaned variables - eg. vidcap doesn't exist if filetype is tif

    #if filetype is avi, read frame-by-frame from avi
    if ftype == ".avi":
        f = os.path.join(wdir,config['infile'])
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

        print("opening .tifs beginning with:", f)

        vidcap=None

    #if filetype is not avi or tif, raise error and end
    else: 
        raise TypeError(f'Filetype {ftype} not recognised')

    print("no frames:", nframes)

    return f, fname, script, spath, wdir, odir, vidcap, flist, nframes, ftype

def getguess(config):
    """
    break out config settings into initial param guess
    """
    guess=np.array([config['amp'], float(config['cs']), config['wl'], config['dz'], config['dm'], config['dec'], config['const'], config['gsig'], config['gamp']])
    return guess

def getbounds(config):
    """
    break out config settings into initial bounds for fit
    """
    bf0=config['bf0']
    bf1=config['bf1']
    bf2=config['bf2']
    bf3=config['bf3']

    bounded=([config['amp']/bf3, float(config['cs'])/(bf0), config['wl']/bf0, config['dz']/bf2, \
        config['dm']/bf2, config['dec']/bf3, config['const']/bf3, config['gsig']/bf2, config['gamp']/bf2], \
        [config['amp']*bf2, float(config['cs'])*bf0, config['wl']*bf0, config['dz']*bf2, config['dm']*bf2, \
        config['dec']*bf2, config['const']*bf1, config['gsig']*bf2, config['gamp']*bf2])

    return bounded

#outputs gaussian with max y = amp
def ngauss(x, mu, sig1, amp):
    g1=norm.pdf(x, mu, sig1)
    g1n=np.divide(g1,max(g1))
    return np.multiply(g1n, amp)


