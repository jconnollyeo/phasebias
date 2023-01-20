import numpy as np
import h5py as h5
import matplotlib.pyplot as plt
# import os
import glob
from datetime import datetime, timedelta
import sys
import argparse
from pathlib import Path

def multilook(im,fa,fr):
    """ 
    Copied this over from the RapidSAR code as I didn't have the rapidsar dir locally and 
    only needed the multilooking while the servers were down.
    
    Averages image over multiple pixels where NaN are present and want to be maintained

    Input:
      im      2D image to be averaged
      fa      number of pixels to average over in azimuth (row) direction
      fr      number of pixels to average over in range (column) direction
    Output:
      imout   2D averaged image
    """
    nr = int(np.floor(len(im[0,:])/float(fr))*fr)
    na = int(np.floor(len(im[:,0])/float(fa))*fa)
    im = im[:na,:nr]
    aa = np.ones((fa,int(na/fa),nr),dtype=im.dtype)*np.nan
    for k in range(fa):
        aa[k,:,:] = im[k::fa,:]
    aa = np.nanmean(aa,axis=0)
    imout=np.ones((fr,int(na/fa),int(nr/fr)),dtype=im.dtype)*np.nan
    for k in range(fr):
        imout[k,:,:] = aa[:,k::fr]
    return np.nanmean(imout,axis=0)

def daisychain(n=360, m=12, startdate="20201204", shape=(1000, 833), wdir="/home/jacob/Satsense/ss2/south_yorkshire", frame_ID="data_with_correction"):
    
    dates = [datetime.strptime(startdate, "%Y%m%d")+timedelta(days=int(d)) for d in np.arange(0, n+1, m)]
    shape = (3000, 10000)
    phase = np.zeros((len(dates), *shape), dtype=float)
    
    phase_mchain_ml = np.zeros((1000, 833), dtype=float)

    for i, (date1, date2) in enumerate(zip(dates[:-1], dates[1:])):

        print (f"{i}/{int(n/m)}")
        date1_str = datetime.strftime(date1, "%Y%m%d")
        date2_str = datetime.strftime(date2, "%Y%m%d")
        phase1_fn = glob.glob(f"{wdir}/{frame_ID}/IFG/singlemaster/*_{date1_str}/*")
        phase2_fn = glob.glob(f"{wdir}/{frame_ID}/IFG/singlemaster/*_{date2_str}/*")

        if len(phase1_fn) == 1:
            phase1 = h5.File(phase1_fn[0])["Phase"][:]
        else:
            sys.exit(f"Error: Bad chain. Date missing: {date1_str}")
        
        if len(phase2_fn) == 1:
            phase2 = h5.File(phase2_fn[0])["Phase"][:]
        else:
            sys.exit(f"Error: Bad chain. Date missing: {date2_str}")

        phase = multilook(np.exp(1j* (phase2 - phase1)), 3, 12)
        
        phase_mchain_ml += np.angle(phase)

    phase000_fn = glob.glob(f"{wdir}/{frame_ID}/IFG/singlemaster/*_{datetime.strftime(dates[0], '%Y%m%d')}/*")
    phase360_fn = glob.glob(f"{wdir}/{frame_ID}/IFG/singlemaster/*_{datetime.strftime(dates[-1], '%Y%m%d')}/*")
    
    print (phase000_fn)
    print (phase360_fn)

    phase000 = h5.File(phase000_fn[0])["Phase"][:]
    phase360 = h5.File(phase360_fn[0])["Phase"][:]

    phase_nchain_ml = multilook(np.exp(1j*(phase360 - phase000)), 3, 12)

    return np.angle(np.exp(1j*(np.angle(phase_nchain_ml) - phase_mchain_ml)))

def coherence_mask(startdate="20201204", wdir="/home/jacob/Satsense/ss2/south_yorkshire/data_with_correction", n=360, m=6, threshold = 0.3):
    
    dates = [datetime.strptime(startdate, "%Y%m%d")+timedelta(days=int(d)) for d in np.arange(0, n+1, m)]
    # mask = np.ones((1000, 833), dtype=bool)
    mask = np.ones((1000, 833), dtype=float)
    
    for i, (date1, date2) in enumerate(zip(dates[:-1], dates[1:])):
        print (f"Coherence: {i}", end="\r")
        
        date1_str = datetime.strftime(date1, "%Y%m%d")
        date2_str = datetime.strftime(date2, "%Y%m%d")

        coh = multilook(h5.File(f"{wdir}/Coherence/{date2_str}/{date1_str}-{date2_str}_coh.h5")["Coherence"][:], 3, 12)
        mask = np.min(np.stack((coh, mask)), axis=0)

        # mask[coh < 0.3] = False
        # mask = mask > 0.3

    return mask

def main():
    
    mask = coherence_mask(threshold=0.2)
    np.save("min_coherence.npy", mask)
    print (f"{np.sum(mask) = }")

    m06 = daisychain(m=6)
    m12 = daisychain(m=12)
    m18 = daisychain(m=18)

    a1 = m12/m06
    a2 = m18/m06
    
    # a1[~mask] = np.nan
    # a2[~mask] = np.nan

    np.save("a_variables_filt.npy", np.stack((a1, a2, mask)))

if __name__ == "__main__":
    sys.exit(main())