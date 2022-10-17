import numpy as np
import h5py as h5
import matplotlib.pyplot as plt
# import os
import glob
from datetime import datetime
import sys
from utils import multilook

wdir = "/workspace/rapidsar_test_data/south_yorkshire/jacob2/"

# Get the lat and lon for each data point

# data = h5.File(wdir + "data.h5")
# shape = np.array(data["Latitude"]).shape
# data = False

# Make a list of all the images
im_path = wdir + 'IFG/singlemaster/*/*.*'

im_fns = [file for file in glob.glob(im_path, recursive=True)]

im_fns.sort() # Sort the images by date

# Use the image fns to get the date of each secondary SLC
dates = [datetime.strptime(d.split('_')[-2], '%Y%m%d') for d in im_fns]

def newIFGs(primary_ix, secondary_ix):

    p1 = np.exp(1j*np.asarray(h5.File(im_fns[primary_ix])['Phase']))
    p2 = np.exp(1j*np.asarray(h5.File(im_fns[secondary_ix])['Phase']))
    
    ifg = p2*p1.conjugate()

    return ifg

def phase_closure(start, end, ml = [3, 12]):
    """
    
    """
    # Getting the dates from the user input indices
    d_ = dates[start:end+1]

    # Performing checks on the dates - do they for a closed loop etc. 
    # assert (d3-d1).days - ((d2-d1).days + (d3-d2).days) == 0, "Does not form closed loop."
    
    shape = multilook(newIFGs(0, 1), ml[0], ml[1]).shape

    # Creating the interferograms 
    ifgs = np.empty((len(d_)-1, *shape), dtype=np.complex64)
    
    print ("Creating interferograms")
    for i, d in enumerate(np.arange(start, end)):
        ifgs[i] = multilook(newIFGs(d, d+1), ml[0], ml[1])

    print ("Creating interferogram spanning whole range")
    ifg_span = multilook(newIFGs(start, end), ml[0], ml[1])
    
    print ("Computing the closure phase")

    closure = ifg_span*np.prod(ifgs, axis=0, dtype=np.complex64).conjugate()
    
    # Formatting a title for the figure
    # loop_dates = f"{datetime.strftime(d1, '%Y%m%d')}_{datetime.strftime(d2, '%Y%m%d')}__\
# {datetime.strftime(d2, '%Y%m%d')}_{datetime.strftime(d3, '%Y%m%d')}__\
# {datetime.strftime(d3, '%Y%m%d')}_{datetime.strftime(d1, '%Y%m%d')}\n{(d2-d1).days}_{(d3-d2).days}_{(d1-d3).days}"
    loop_days = [str((dates[di+1] - dates[di]).days) for di in np.arange(start, end, 1)]
    loop_dates = f"{(dates[end]-dates[start]).days} - (" + ', '.join(loop_days) + f") | ML = [{ml[0]}, {ml[1]}] \n {dates[start].strftime('%Y%m%d')} to {dates[end].strftime('%Y%m%d')}"

    # loop_dates = f"n={(dates[end]-dates[start]).days}, m={6}"

    # Returning the phase closure in radians and the figure title
    return np.angle(closure), loop_dates

def plot_phase_closure(arr, loop_dates, ml): 
    """ 
    
    """
    fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(12, 4), gridspec_kw=dict(width_ratios=[1/3, 1/3, 1/3]))
    amp = multilook(np.load("/home/jacob/phase_bias/mean_amp.npy"), ml[0], ml[1])
    ax[0].matshow(amp, vmax=3, cmap="binary_r")
    p = ax[1].matshow(arr, cmap='RdYlBu', vmin=-np.pi, vmax=np.pi)
    n = ax[2].hist(arr.flatten(), bins=np.linspace(-np.pi, np.pi, 50), alpha=0.8)[0]
    ax[2].text(-0.5, np.max(n)*0.7, f"$\mu$ = {np.mean(arr):.2f}", horizontalalignment='right')
    ax[2].axvline(x=0, color='black')
    
    ax[0].get_shared_x_axes().join(ax[0], ax[1])
    for a in ax:
        a.grid()

    ax[0].set_title("Mean amplitude")
    ax[1].set_title("Closure phase")
    ax[2].set_title("Closure phase histogram")
    
    plt.suptitle(loop_dates, y=0.08)
    # print (p)
    cbar = plt.colorbar(p, ax=ax[2], orientation='horizontal')
    cbar.ax.set_ylabel("Closure phase (radians)")
    # cbar.ax.set_ylim([-np.pi, np.pi])
    # cbar.ax.set_yticks([-np.pi, 0, np.pi])
    # cbar.ax.set_yticklabels(["$-\pi$", "0", "$\pi$"])
    plt.show()

def main():

    start, end, ml1, ml2 = np.array(sys.argv[1:], dtype=int)

    arr, loop_dates = phase_closure(start, end, [ml1, ml2])

    plot_phase_closure(arr, loop_dates, [ml1, ml2]) # [447:691, 31:810]

if __name__ == "__main__":
    main()