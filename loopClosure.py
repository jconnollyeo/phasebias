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
# lat = data["Latitude"]
# lon = data["Longitude"]

# Make a list of all the images
im_path = wdir + 'IFG/singlemaster/*/*.*'

im_fns = [file for file in glob.glob(im_path, recursive=True)]

im_fns.sort() # Sort the images by date

# Use the image fns to get the date of each secondary SLC
dates = [datetime.strptime(d.split('_')[-2], '%Y%m%d') for d in im_fns]

# days = [(dates[i+1]-dates[i]).days for i in range(len(dates)-1)]
# plt.plot(days, np.ones(len(days)), 'b.')
# plt.show()

def phase_closure(i1, i2, i3, ml = [3, 12]):
    """
    
    """
    # Getting the dates from the user input indices
    d1, d2, d3 = dates[i1], dates[i2], dates[i3]

    # Performing checks on the dates - do they for a closed loop etc. 
    assert (d3-d1).days - ((d2-d1).days + (d3-d2).days) == 0, "Does not form closed loop."
    # assert (d2-d1).days == 12, f"d1 and d2 should be 12 days apart. {(d2-d1).days}"
    # assert (d3-d2).days == 12, f"d2 and d3 should be 12 days apart. {(d3-d2).days}"
    # assert (d1-d3).days == -24, f"d3 and d1 should be -24 days apart. {(d1-d3).days}"

    # Converting the phase in radians into complex form. 
    p1 = np.exp(1j*np.asarray(h5.File(im_fns[i1])['Phase']))
    p2 = np.exp(1j*np.asarray(h5.File(im_fns[i2])['Phase']))
    p3 = np.exp(1j*np.asarray(h5.File(im_fns[i3])['Phase']))

    # Multilooking the phase
    # p1 = multilook(p1, ml[0], ml[1])
    # p2 = multilook(p2, ml[0], ml[1])
    # p3 = multilook(p3, ml[0], ml[1])
    
    # Creating the interferograms 
    ifg12 = p2*p1.conjugate()
    ifg23 = p3*p2.conjugate()
    ifg13 = p3*p1.conjugate()

    # Multilook the new interferograms
    ifg12 = multilook(ifg12, ml[0], ml[1])
    ifg23 = multilook(ifg23, ml[0], ml[1])
    ifg13 = multilook(ifg13, ml[0], ml[1])
    
    # Computing the closure phase
    closure = ifg13*(ifg12*ifg23).conjugate()
    
    # Formatting a title for the figure
    loop_dates = f"{datetime.strftime(d1, '%Y%m%d')}_{datetime.strftime(d2, '%Y%m%d')}__\
{datetime.strftime(d2, '%Y%m%d')}_{datetime.strftime(d3, '%Y%m%d')}__\
{datetime.strftime(d3, '%Y%m%d')}_{datetime.strftime(d1, '%Y%m%d')}\n{(d2-d1).days}_{(d3-d2).days}_{(d1-d3).days}"
    
    # Returning the phase closure in radians and the figure title
    return np.angle(closure), loop_dates

def plot_phase_closure(arr, loop_dates): 
    """ 
    
    """
    fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(12, 6))
    amp = multilook(np.load("/home/jacob/phase_bias/mean_amp.npy"), 3, 12)
    ax[0].matshow(amp, vmax=3, cmap="binary_r")
    p = ax[1].matshow(arr, cmap='RdYlBu', vmin=-np.pi, vmax=np.pi)
    ax[2].hist(arr.flatten(), bins=np.linspace(-np.pi, np.pi, 50), alpha=0.8)
    ax[2].axvline(x=0, color='black')

    ax[0].get_shared_x_axes().join(ax[0], ax[1])
    
    plt.suptitle(loop_dates)
    # print (p)
    plt.colorbar(p, ax=ax[1])
    # cbar.ax.set_ylim([-np.pi, np.pi])
    # cbar.ax.set_yticks([-np.pi, 0, np.pi])
    # cbar.ax.set_yticklabels(["$-\pi$", "0", "$\pi$"])

    plt.show()

def main():

    i1, i2, i3, ml1, ml2= np.array(sys.argv[1:], dtype=int)

    arr, loop_dates = phase_closure(i1, i2, i3, [ml1, ml2])

    plot_phase_closure(arr, loop_dates)

if __name__ == "__main__":
    main()