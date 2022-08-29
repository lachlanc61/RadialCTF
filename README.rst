Simple tool to fit radial contrast transfer function to TEM images to detect radial drift in first zero

- Receives either .avi movie files or stacks of .tif files
- Calculates FFT if in real space
- Generates radial profiles around image centre at given phi, sector width
- Fits CTF to each sector profile
- calculates basic properties across stack

Run via core.py
- global variables at beginning of core.py control eg. filetype and other params

Example data in ./data
