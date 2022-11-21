import numpy as np
import matplotlib.pyplot as plt
from loopClosure import getFiles, create_loop
from loopClosure_post import splitGrids, importData, extractDates
import sys
from datetime import datetime, timedelta
import argparse
import h5py as h5
from utils import multilook
from statistics import mode
import glob

def parse_args():
    """
    Parses command line arguments and defines help output
    """
    parser = argparse.ArgumentParser(description='Compute and plot the closure phase.')
    parser.add_argument("-w",
                        dest="wdir", 
                        type=str,
                        help='Directory containing the triplet loop closure.',
                        default='/workspace/rapidsar_test_data/south_yorkshire/jacob2/IFG/singlemaster/*/*.*')
    parser.add_argument("-i",
                        type=int,
                        dest='start',
                        default=286,
                        help='Start index of the loop')
    parser.add_argument("-l",
                        type=int,
                        dest='length',
                        default=360,
                        help='Length of the loop (days)')
    parser.add_argument("-m",
                        type=int,
                        dest='m',
                        default=6,
                        help='length of m-day ifgs (the shorter ones)')
    parser.add_argument("-n",
                        type=int,
                        dest='n',
                        default=60,
                        help='length of n-day ifgs (the longer ones)')
    parser.add_argument("-s",
                        type=bool,
                        dest="save",
                        help="Boolean to save (True) or not save (False).",
                        default=0)
    parser.add_argument("-p",
                        type=bool,
                        dest="plot",
                        help="Boolean to show plot (True) or not show plot (False).",
                        default=1)
    parser.add_argument("-a",
                        type=str,
                        dest='multilook',
                        help="Multilook factor",
                        default='3,12')
    parser.add_argument("-g",
                        type=int,
                        dest='grid',
                        help="Segment into grids",
                        default=0)
    args = parser.parse_args()
    return vars(args)

def multilook_mode(arr, ml, preserve_res=True):
    start_indices_i = np.arange(arr.shape[0])[::ml[0]]
    start_indices_j = np.arange(arr.shape[1])[::ml[1]]
    
    out = np.zeros(arr.shape, dtype=arr.dtype)

    for i in start_indices_i:
        for j in start_indices_j:
            out[i:i+ml[0], j:j+ml[1]] = mode(arr[i:i+ml[0], j:j+ml[1]].flatten())

    if preserve_res:
        return out
    else:
        end_i = out.shape[0] % ml[0]
        end_j = out.shape[1] % ml[1]

        if end_i == 0:
            end_i = None
        else:
            end_i = -1*end_i

        if end_j == 0:
            end_j = None
        else:
            end_j = -1*end_j

        return out[:end_i:ml[0], :end_j:ml[1]]

def convert_landcover(im):
    """
    Keeps the urban,  classification the same.

    Args:
        im (_type_): _description_

    Returns:
        _type_: _description_
    """
    out = im.copy()

    out[np.logical_and(im >= 111, im <= 126)] = 111 # Forests
    out[np.logical_or(im == 90, im == 80)] = 200 # Water
    out[np.logical_or(im == 70, im == 200)] = 200 # Water
    out[np.logical_or(im == 20, im == 30)] = 20 # shrubs, herb veg, moss
    out[np.logical_or(im == 20, im == 100)] = 20 

    return out

def landcover(arr, fp, ml):
    """
    Args:
        arr (3d array): 3d array of the loop closure. Along the zeroth axis, each 2d array is of 
                        a different m-day interferogram loop. 
    """
    types = {111:"Forest", 20: "Shrubs", 40:"Cropland", 50:"Urban", 200:"Water/Ice", 30:"Herbacious Veg.", 100:"Moss", }
    lc = h5.File(fp)["Landcover"]
    lc_conv = convert_landcover(np.array(lc))
    lc_ml = multilook_mode(lc_conv, ml, preserve_res=False)
    print (lc_ml.shape)
    
    lc_loop = np.empty((arr.shape[0], 4))
    for i, im in enumerate(arr):
        for l, lc_type in enumerate(np.array([111, 50, 40, 20])):
            # print (np.sum(lc_ml == lc_type))
            lc_loop[i, l] = np.angle(np.mean(np.exp(1j*im)[lc_ml == lc_type]))
    
    return lc_loop, {111:"Forest", 50:"Urban", 40:"Cropland", 20:"Shrubs"} # [types[111], types[50], types[40], types[20]]

def main():
    plt.rcParams['image.cmap'] = 'RdYlBu'
    plt.style.use(['seaborn-poster'])    

    # Go through files and make a chain of t small n-length loops.

    fp = '/home/jacob/phase_bias/12_6_6/data/*.npy'
    data_fns = [file for file in glob.glob(fp, recursive=True)]
    data_fns.sort()
    dates = extractDates(data_fns)
    # data = importData(data_fns)
    # data_complex = np.exp(1j*data)
    shape = np.load(data_fns[0]).shape
    datatype = np.load(data_fns[0]).dtype

    start_ix = 0
    ts = range(1, 21)
    out = np.empty((len(ts), *shape), dtype=datatype)

    fig = plt.figure(figsize=(12, 15))
    ax = np.array([fig.add_subplot(351),
    fig.add_subplot(352),
    fig.add_subplot(353),
    fig.add_subplot(354),
    fig.add_subplot(355),
    fig.add_subplot(356),
    fig.add_subplot(357),
    fig.add_subplot(358),
    fig.add_subplot(359),
    fig.add_subplot(3, 5, 10),
    fig.add_subplot(313)])

    grid = True

    for t in ts:
        t_out = np.zeros((t, *shape))
        t_fns = data_fns[start_ix::2][:t+1]

        for fn in t_fns:
            t_out[t-1] += np.load(fn)
        t_sum = np.sum(t_out, axis=0)

        if t % 2 == 0:    
            if grid:
                p = ax[int((t/2)-1)].matshow(np.angle(splitGrids(np.exp(1j*t_sum), size=40)), vmin=-5, vmax=5)
            else:
                p = ax[int((t/2)-1)].matshow(t_sum, vmin=-5, vmax=5)
            ax[int((t/2)-1)].axis('off')
            ax[int((t/2)-1)].set_title(t)
        out[t-1] = t_sum

    closure_all = np.array([np.mean(im) for im in out])  # Changed to mean instead of sum
    print (t_out.shape)

    ax[-1].plot(range(1, 21, 1), np.cumsum(closure_all), label="All pixels", color="black")
    landcover_to_plot = np.empty((4, closure_all.shape[0]))

    landcover_closure, types = landcover(out, "/workspace/rapidsar_test_data/south_yorkshire/datacrop_20220204.h5", ml=[3,12])
    for l, type in enumerate(types.keys()):
        ax[-1].plot(range(1, 21, 1), np.cumsum(landcover_closure[:, l]), label=types[type])

    ax[-1].set_xticks(np.arange(1, 21, 1), np.arange(1, 21, 1))

    plt.colorbar(p, ax=ax[:-1], shrink=0.8)
    plt.legend()
    ax[-1].set_xlabel("$t$")
    ax[-1].set_ylabel("$\sum^{t}_{i=1} (\Delta\Sigma ^{18-6})_i$")
    plt.show()

if __name__ == "__main__":
    sys.exit(main())