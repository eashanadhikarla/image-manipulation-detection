"""
Name    : Eashan Adhikarla
Subject : Media Forensics
Project : Mini-project-2 (Task 2)
Data    : April 10, 2021

"""

import os, cv2, glob, pickle
import numpy as np
from matplotlib import pyplot as plt


# rootdir = "/Users/eashan22/Dropbox (LU Student)/Macbook/Desktop/Media Forensics/mini-project-2/Task 2/"
rootdir = "/data/MediaForensics/DeepFake/Frequency/"

path    = ['Faces-HQ/thispersondoesntexists_10K',
           'Faces-HQ/100KFake_10K',
           'Faces-HQ/Flickr-Faces-HQ_10K',
           'Faces-HQ/celebA-HQ_10K',
           ]

datadir = "./data/FacesHQ_Data.pkl"


labels = [1, 1, 0, 0]
epsilon = 1e-8
Data = {}


# Number of samples from each dataset
iter_ = 0
dataLimit = 500
samples = 4 * dataLimit
Azimuthalavg1D = np.zeros([samples, 722])
label_total = np.zeros([samples])


def AzimuthalAverage(image, center=None):
    """
    Calculate the azimuthally averaged radial profile.

    image  - The 2D image
    center - The [x,y] pixel coordinates used as the center. The default is 
             None, which then uses the center of the image (including 
             fracitonal pixels).
    
    """
    # Calculate the indices from the image
    y, x = np.indices(image.shape)

    if not center:
        center = np.array([(x.max()-x.min())/2.0, (y.max()-y.min())/2.0])

    r = np.hypot(x - center[0], y - center[1])

    # Get sorted radii
    ind = np.argsort(r.flat)
    r_sorted = r.flat[ind]
    i_sorted = image.flat[ind]

    # Get the integer part of the radii (bin size = 1)
    r_int = r_sorted.astype(int)

    # Find all pixels that fall within each radial bin.
    deltar = r_int[1:] - r_int[:-1]  # Assumes all radii represented
    rind = np.where(deltar)[0]       # location of changed radius
    nr = rind[1:] - rind[:-1]        # number of radius bin
    
    # Cumulative sum to figure out sums for each radius bin
    csim = np.cumsum(i_sorted, dtype=float)
    tbin = csim[rind[1:]] - csim[rind[:-1]]

    radial_prof = tbin / nr

    return radial_prof


def preprocessingData(iter_, dataLimit, samples, Azimuthalavg1D, label_total):
    """
    Step 1. DTF 
    Step 2. Amplitude Spectrum 2D
    Step 3. Azimuthal Averaged
    Step 4. Azimuthal Spectrum 1D

    """
    for data in range(len(path)):
        dataIdxCount = 0
        psd1D_average_org = np.zeros(722)
        print(f"Processing dataset {path[data]}...")

        for filename in glob.glob(str(rootdir)+path[data]+"/*.jpg"):
            # print(filename)
            
            # Read every image in the root directory
            img = cv2.imread(filename, 0)
            
            # To compute the 2-dimensional discrete Fourier Transform
            f = np.fft.fft2(img)
            
            # Shift the zero-frequency component to the center of the spectrum
            fshift = np.fft.fftshift(f)
            fshift += epsilon

            magnitude_spectrum = 20 * np.log(np.abs(fshift))

            # Calculate the Azimuthal averaged 1D power spectrum
            psd1D = AzimuthalAverage(magnitude_spectrum)
            Azimuthalavg1D[iter_, :] = psd1D
            label_total[iter_] = labels[data]

            dataIdxCount += 1
            iter_ += 1
            if dataIdxCount >= dataLimit:
                break
    return Azimuthalavg1D, label_total

# Azimuthalavg1D, label_total = preprocessingData(iter_, dataLimit, samples, Azimuthalavg1D, label_total)
# print(f"Total images: {len(Azimuthalavg1D)}\nTotal labels: {len(label_total)}")
# Data["data"], Data["label"] = Azimuthalavg1D, label_total
# print(label_total)
# output = open(datadir, 'wb')
# pickle.dump(Data, output)
# output.close()
# print("Data Preprocessed and Saved")




















