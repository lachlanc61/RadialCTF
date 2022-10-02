# Overview

This tool is developed to monitor the contrast transfer function (CTF) of a transmission electron microscope (TEM) during operation. These instruments use a highly-focused electron beam and a series of electromagnetic lenses to perform nanometer-scale imaging, at resolutions far exceeding those possible with an optical microscope. 

Unsurprisingly, operation is highly technical. A particularly challenging aspect is the selection of the "defocus" parameter; novice operators will tend to maximise contrast at the expense of detail, reducing the spatial-resolution of the image. Lens drift may also lead to asymmetric effects, which can be difficult for an inexperienced operator to identify.

This tool is designed to assist with these challenges. It processes a time-series from the microscope, and performs fourier transformation (if needed), radial integration, and fitting of the contrast-transfer function. This produces simple metrics reporting the defocus offset and anisotropy of the image, which can be monitored during measurement to assist the operator.  

# Summary:

- parses stacked greyscale .tif images or movies

- performs a fast fourier transform, converting real-space images to frequency-space

- applies series of radial masks to evaluate anisotropy

- performs azimuthal averaging to obtain profiles of intensity vs frequency

- fits the analytical contrast transfer function to these profiles

- exports plots of sectors and fitted functions

- reports goodness of fit, position of first minimum, and variance in first minimum with angle

# Method

The data is....

- fileformat...
    - (src.radial)

<p align="left">
  <img src="./docs/IMG/fileformat4.png" alt="Spectrum" width="1024">
  <br />
</p>

#

<p align="left">
  <img src="./docs/IMG/hsv_spectrum2.png" alt="Spectrum" width="700">
  <br />
</p>

<p align="left">
  <img src="./docs/IMG/geo_colours2.png" alt="Spectrum" width="700">
  <br />
</p>

#

<p align="left">
  <img src="./docs/IMG/geo_clusters.png" alt="Spectrum" width="1024">
  <br />
</p>

# Usage

The tool is run as a script from core.py, or Jupyter notebook explore.ipynb

An example dataset is provided in ./data

The path to the dataset to be analysed is set in config.py, together with various flags and control parameters. 
