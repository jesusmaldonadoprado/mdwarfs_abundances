################################################################
# README file          					       #	
# version 1.0: Wed Sep 16 11:58:03 CEST 2020                   #
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#
# See Maldonado et al. (2020, submitted to A&A) for details.   #
# Please report any bug or suggestion to:                      #
# jesus.maldonado@inaf.it                                      #
################################################################

O. System requirements:
------------------------------------------------------
This code requires the use of several python packages:
matplotlib
sklearn
PyAstronomy
scipy
pandas
numpy
statistics
pylab

**** NOTE ****
This code was developed for HARPS / HARPS-N spectra.
We caution that our methods are untested or unreliable
on other spectrographs.


1. Files description:   
------------------------------------------------------ 
- "README.txt" 
  This file

- "mdwarfs_pca_metallicity_analysis_full_code_ver_1.0.py"
  Python code to derive the stellar abundances of M dwarfs

- "MDWARFS_metal_ifile_for_pca_ver1.0.csv"
  File containing the spectra of the M dwarf stars in the
  training dataset and those problem stars

- "MDWARFS_metal_calibration_data_all_elements_ver1.0.csv"
  File containing the stellar abundances for the M dwarfs
  in the training dataset

- "fits" directory 
  It contains the  HARPS/HARPS-N spectra
  of the M dwarfs in the training dataset and several
  examples of "problem" stars

- "MDWARFS_abundances_with_indexes_pca_all_elements_example_output.txt"
  Example of an output file


2. How to use the code:
------------------------------------------------------
2.1 Edit the "mdwarfs_pca_metallicity_analysis_full_code_ver_1.0.py"
    file:

Open this file and set the following variables:

"spectra_dir (line 42)": write the directory of the folder containing
both the training and problem stars spectra

"ofile (line 45)": write the name that you want for your output file

You do not need to change anything else.

2.2. Edit the file "MDWARFS_metal_ifile_for_pca_ver1.0.csv" 
After the spectra of the training stars, add your "problem" stars
(one per line, .fits format)

2.3 Run the python program

3. Outputs:
------------------------------------------------------  
The output file includes:
- A list of the stars used in the training dataset
- The derived abundances for the stars in the training dataset
(one star per line, same order as in the ifile)
- The derived abundances for the "problem" stars
(one star per line, same order as in the ifile) 


