from ast import arg
import numpy as np
import h5py as h5
import matplotlib.pyplot as plt
# import os
import glob
from datetime import datetime
import sys
from utils import multilook
import argparse
from pathlib import Path


# wdir = "/workspace/rapidsar_test_data/south_yorkshire/jacob2/"

def newIFGs(primary_ix, secondary_ix, im_fns):
    """
    Produce a new ifgs spanning the two indices from h5 files using index
    in wdir. 
    """
    p1 = np.exp(1j*np.asarray(h5.File(im_fns[primary_ix])['Phase']))
    p2 = np.exp(1j*np.asarray(h5.File(im_fns[secondary_ix])['Phase']))
    
    ifg = p2*p1.conjugate()

    return ifg

def phase_closure(start, end, dates, im_fns, ml = [3, 12]):
    """
    Produce the phase closure of the defined indices. This will be a series 
    of ifgs of the shortest possible baseline and one spanning the entire time. 
    """
    # Getting the dates from the user input indices
    d_ = dates[start:end+1]

    shape = multilook(newIFGs(0, 1, im_fns), ml[0], ml[1]).shape

    # Creating the interferograms 
    ifgs = np.empty((len(d_)-1, *shape), dtype=np.complex64)
    
    print ("Creating interferograms")
    for i, d in enumerate(np.arange(start, end)):
        ifgs[i] = multilook(newIFGs(d, d+1, im_fns), ml[0], ml[1]) # Complex np.complex64

    print ("Creating interferogram spanning whole range")
    ifg_span = multilook(newIFGs(start, end, im_fns), ml[0], ml[1]) # Complex np.complex64
    
    print ("Computing the closure phase")
    closure = ifg_span*np.prod(ifgs, axis=0, dtype=np.complex64).conjugate() # Phi_start, end - (Phi_start,start+1 + Phi_start+1,start+2, etc.)
    
    # Formatting a title for the figure
    loop_days = [str((dates[di+1] - dates[di]).days) for di in np.arange(start, end, 1)]
    loop_dates = f"{(dates[end]-dates[start]).days} - (" + ', '.join(loop_days) + f") | ML = [{ml[0]}, {ml[1]}] \n\
        {dates[start].strftime('%Y%m%d')} to {dates[end].strftime('%Y%m%d')}"

    # Returning the phase closure in radians and the figure title
    return np.angle(closure), loop_dates

def plot_phase_closure(arr, loop_dates, dates, ml, save, plot): 
    """ 
    Func to plot the closure phase in radar coords and as a histogram. The mean amplitude is also plotted in radar coords. 
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
    cbar = plt.colorbar(p, ax=ax[2], orientation='horizontal')
    cbar.ax.set_ylabel("Closure phase (radians)")

    save_fn = f"12_6_6/ML12_48/{loop_dates.split(' ')[-3]}_{loop_dates.split(' ')[-1]}.png"
    if save:
        plt.savefig(save_fn)
        np.save(save_fn[:-4] + ".npy", arr)
    else:
        pass
    if plot:
        plt.show()
    else:
        pass

def parse_args():
    """
    Parses command line arguments and defines help output
    """
    parser = argparse.ArgumentParser(description='Compute and plot the closure phase.')
    parser.add_argument("-w",
                        dest="wdir", 
                        type=str,
                        help='Directory containing the single master interferogram phase hdf5 files',
                        default="/workspace/rapidsar_test_data/south_yorkshire/jacob2/")
    parser.add_argument("-i",
                        type=int,
                        dest='start_index',
                        default=0,
                        help='Start index of the loop')
    parser.add_argument("-d",
                        type=int,
                        dest='loop_delta',
                        default=2,
                        help='Delta for the jumpy (ie. 2 for a triplet.')
    parser.add_argument("-m",
                        type=str,
                        dest='multilook',
                        help="Multilook factor",
                        default='3,12')
    parser.add_argument("-s",
                        type=bool,
                        dest="save",
                        help="Boolean to save (True) or not save (False).",
                        default=1)
    parser.add_argument("-p",
                        type=bool,
                        dest="plot",
                        help="Boolean to show plot (True) or not show plot (False).",
                        default=1)
    args = parser.parse_args()
    return vars(args)

def main():
    args_dict = parse_args()
    wdir = str(args_dict["wdir"])
    start = int(args_dict["start_index"])
    delta = int(args_dict["loop_delta"])
    ml = np.array(args_dict["multilook"].split(','), dtype=int)
    save_bool = args_dict["save"]
    plot = args_dict["plot"]
    # Make a list of all the images
    im_path = wdir + 'IFG/singlemaster/*/*.*'

    im_fns = [file for file in glob.glob(str(im_path), recursive=True)]

    im_fns.sort() # Sort the images by date

    # Use the image fns to get the date of each secondary SLC
    dates = [datetime.strptime(d.split('_')[-2], '%Y%m%d') for d in im_fns]

    # start, delta, ml1, ml2 = np.array(sys.argv[1:], dtype=int)
    end = int(start) + int(delta)
    arr, loop_dates = phase_closure(start, end, dates, im_fns, ml)

    plot_phase_closure(arr, loop_dates, dates, ml, save_bool, plot)

if __name__ == "__main__":
    sys.exit(main())