import numpy as np
import matplotlib.pyplot as plt
from loopClosure import getFiles, create_loop
from loopClosure_post import splitGrids
import sys
from datetime import datetime, timedelta
import argparse
import h5py as h5
from utils import multilook
from statistics import mode

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

def check_chain(dates, length):
    d = dates[0]
    ixs = [0]

    while not (d == dates[-1]):
        if d+timedelta(days=length) in dates:
            ixs.append(int(np.where((np.array(d+timedelta(days=length)) == dates))[0][0]))
            d = d+timedelta(days=length)
        else:
            print (d, length)
            raise Exception(f"Not a full closed loop: {length = }, {d = }, {d+timedelta(days=length) = }\n\n {dates}")
    print (ixs)
    return True, ixs


def main():
    plt.rcParams['image.cmap'] = 'RdYlBu'

    args_dict = parse_args()
    wdir = str(args_dict["wdir"])
    start = int(args_dict["start"])
    length = int(args_dict["length"])
    m = int(args_dict["m"])
    n = int(args_dict["n"])
    ml = np.array(args_dict["multilook"].split(','), dtype=int)
    grid = bool(args_dict["grid"])
    fns = getFiles(wdir)

    save_bool = args_dict["save"]
    plot = args_dict["plot"]
    if m:
        n_loop = doLoops(fns, start, length, delta=n, ml=ml) # 60 day loop for 360 days

        m_loop = doLoops(fns, start, length, delta=m, ml=ml) # 60 day loop for 360 days
        diff = np.angle(np.exp(1j*(n_loop - m_loop)))    

        plt.matshow(diff)
        plt.colorbar()
    else:
        shape = multilook(h5.File(fns[0])['Phase'][:], ml[0], ml[1]).shape
        out = np.empty((6, *shape))
        
        fig = plt.figure(figsize=(12, 15))
        ax = np.array([fig.add_subplot(331),
        fig.add_subplot(332),
        fig.add_subplot(333),
        fig.add_subplot(334),
        fig.add_subplot(335),
        fig.add_subplot(336),
        fig.add_subplot(313)])
        
        n_loop = doLoops(fns, start, length, delta=n, ml=ml) # 60 day loop for 360 days

        for a, m in zip(ax[:6], range(6, 37, 6)):
            print (f"{n = }, {m = }")
            m_loop = doLoops(fns, start, length, delta=m, ml=ml)

            lc = np.angle(np.exp(1j*(n_loop - m_loop)))

            out[int((m/6)-1)] = lc
            if grid:
                lc = np.angle(splitGrids(np.exp(1j*lc), size=40))
            else:
                print (f"{grid = }")
            a.axis('off')
            
            p = a.matshow(lc, vmax=np.pi, vmin=-np.pi)
            a.set_title(f"{m = }")

        np.save("loop_closure.npy", out)
        plt.colorbar(p, ax=ax[:6], shrink=0.8)
        lc_loop, labels = landcover(out, "/workspace/rapidsar_test_data/south_yorkshire/datacrop_20220204.h5", ml)

        for i, l in enumerate(lc_loop.T):
            ax[-1].plot(np.arange(6, 37, 6), l, label=labels[i])
        ax[-1].set_xticks(np.arange(6, 37, 6), np.arange(6, 37, 6).astype(str))
        ax[-1].legend()

    if save_bool:
        plt.savefig("test.jpg")
    else:
        pass

    if plot:
        plt.show()


def doLoops(fns, start, length, delta, ml):
    # === Description ===

    # The 60 days span the entire 360 days, then the shorter 
    # interferograms also span the same time but they form on 
    # closed loop, not lots daisy chained together
    
    # === Plan ===

    # Isolate the full time series to be used for the loop
    dates = [datetime.strptime(d.split('_')[-2], '%Y%m%d') for d in fns]
    print (len(dates))
    end = np.where(np.array([(d - dates[start]).days for d in dates]) == length)[0]
    if len(end) == 0:
        sys.exit("Data does not allow for loop of specified size")
    else:
        end = int(end[0])

    # print (dates[end])
    # Check that chains of n and m day ifgs can be created
    delta_chain_check, delta_ixs = check_chain(dates[start:end+1], length=delta)
    print (f"Checked {delta = }")

    # Create daisy chain of n-day ifgs (60 days)
    delta_days = np.arange(0, length+1, delta) # [0, 60, 120, 180, ..., 360]  
    shape = multilook(h5.File(fns[0])['Phase'][:], ml[0], ml[1]).shape # Fetch the shape of the data
    delta_ifgs_summed = np.zeros(shape) 

    for p_fn, s_fn in zip(delta_ixs[:-1], delta_ixs[1:]): 
        print (f"delta, {p_fn = }, {s_fn = }", end='\r')
        p = h5.File(fns[p_fn])['Phase'][:] 
        s = h5.File(fns[s_fn])['Phase'][:] 

        delta_ifgs_summed += np.angle(multilook(np.exp(1j*(p-s)), ml[0], ml[1]))
    
    return delta_ifgs_summed

def multilook_mode(arr, ml, preserve_res=True):
    """_summary_

    Args:
        arr (2d array): Array to multilook
        ml (len=2 list): Multilook factor (fa, fr)
        preserve_res (bool, optional): This forces the size of the output to be the same as the input - each 
        multilooked cell will be of size (fa, fr). Defaults to True.

    Returns:
        2d array: multilooked array
        perc (2d array): Percentage of the pixels that are the same as the mode pixel
    """

    start_indices_i = np.arange(arr.shape[0])[::ml[0]]
    start_indices_j = np.arange(arr.shape[1])[::ml[1]]
    
    out = np.zeros(arr.shape, dtype=arr.dtype)

    perc = np.zeros(arr.shape, dtype=float)

    for i in start_indices_i:
        for j in start_indices_j:
            m = mode(arr[i:i+ml[0], j:j+ml[1]].flatten())
            out[i:i+ml[0], j:j+ml[1]] = m
            p = np.sum(arr[i:i+ml[0], j:j+ml[1]] == m)/\
                        (out[i:i+ml[0], j:j+ml[1]].size)

            perc[i:i+ml[0], j:j+ml[1]] = p

    if preserve_res:
        return out, perc
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

        out = out[:end_i:ml[0], :end_j:ml[1]]
        perc = perc[:end_i:ml[0], :end_j:ml[1]]

        return out, perc

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
    types = {111:"Forest", 20: "Shrubs", 40:"Cropland", 50:"Urban", 200:"Water/Ice", 30:"Herbacious Veg.", 100:"Moss"}
    lc = h5.File(fp)["Landcover"]
    lc_conv = convert_landcover(lc[:])
    lc_ml, perc = multilook_mode(lc_conv, ml, preserve_res=False)

    perc_mask = perc == 1

    print (lc_ml.shape)

    mask = multilook(np.load("mean_coherence_20201228_20211217.npy"), ml[0], ml[1]) > 0.3
    
    lc_loop = np.empty((arr.shape[0], 4))
    for i, im in enumerate(arr):
        for l, lc_type in enumerate(np.array([111, 50, 40, 20])):
            # print (np.sum(lc_ml == lc_type))
            lc_loop[i, l] = np.angle(np.mean(np.exp(1j*im)[(lc_ml == lc_type) & mask])) # & perc_mask]))
    
    return lc_loop, [types[111], types[50], types[40], types[20]]

if __name__ == "__main__":
    sys.exit(main())