import numpy as np
import matplotlib.pyplot as plt
from loopClosure import getFiles, create_loop
from loopClosure_post import splitGrids
import sys
from datetime import datetime, timedelta
import argparse
import h5py as h5
from utils import multilook

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
                        type=bool,
                        dest='grid',
                        help="Segment into grids",
                        default=False)
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
            print (d)
            return False, []
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
    grid = args_dict["grid"]

    fns = getFiles(wdir)

    save_bool = args_dict["save"]
    plot = args_dict["plot"]

    if m:
        diff = doLoops(fns, start, length, m, n, ml)
        plt.matshow(diff)
        plt.colorbar()
    else:
        shape = multilook(np.array(h5.File(fns[0])['Phase']), ml[0], ml[1]).shape
        # shape = doLoops(fns, start, length, 6, n, ml).shape
        out = np.empty((6, *shape))
        
        # fig, ax = plt.subplots(nrows=2, ncols=3, figsize=(12, 10), sharex=True, sharey=True)
        fig = plt.figure(figsize=(12, 15))
        ax = np.array([fig.add_subplot(331),
        fig.add_subplot(332),
        fig.add_subplot(333),
        fig.add_subplot(334),
        fig.add_subplot(335),
        fig.add_subplot(336),
        fig.add_subplot(313)])

        for a, m in zip(ax[:6], range(6, 37, 6)):
        # for a, m in zip([ax[0]], [36]):
            print (f"{n = }, {m = }")
            lc = doLoops(fns, start, length, m, n, ml)
            out[int((m/6)-1)] = lc
            if grid:
                lc = np.angle(splitGrids(np.exp(1j*lc), size=40))
            else:
                pass
            
            p = a.matshow(lc, vmax=np.pi, vmin=-np.pi)
            a.set_title(f"{m = }")
        np.save("loop_closure.npy", out)
        plt.colorbar(p, ax=ax[:6], shrink=0.8)
        lc_loop, labels = landcover(out, "/workspace/rapidsar_test_data/south_yorkshire/datacrop_20220204.h5", ml)
        # print (lc_loop.shape)
        # print (len(labels))
        for i, l in enumerate(lc_loop.T):
            ax[-1].plot(np.arange(6, 37, 6), l, label=labels[i])
        # ax[-1].plot(np.arange(6, 37, 6), lc_loop, label=labels)
        ax[-1].set_xticks(np.arange(6, 37, 6), np.arange(6, 37, 6).astype(str))
        ax[-1].legend()
    if save_bool:
        plt.savefig("test.jpg")
    else:
        pass

    if plot:
        plt.show()


def doLoops(fns, start, length, m, n, ml):
    # === Description ===

    # The 60 days span the entire 360 days, then the shorter 
    # interferograms also span the same time but they form on 
    # closed loop, not lots daisy chained together
    
    # === Plan ===

    # Isolate the full time series to be used for the loop
    dates = [datetime.strptime(d.split('_')[-2], '%Y%m%d') for d in fns]
    end = np.where(np.array([(d - dates[start]).days for d in dates]) == length)[0]
    if len(end) == 0:
        sys.exit("Data does not allow for loop of specified size")
    else:
        end = int(end[0])

    # print (dates[end])
    # Check that chains of n and m day ifgs can be created
    n_chain_check, n_ixs = check_chain(dates[start:end+1], length=n)
    print ("Checked n")
    m_chain_check, m_ixs = check_chain(dates[start:end+1], length=m)
    print ("Checked m")
    # Calculate the days

    # Create daisy chain of n-day ifgs (60 days)
    n_days = np.arange(0, length+1, n) # [0, 60, 120, 180, ..., 360]  
    shape = multilook(np.asarray(h5.File(fns[0])['Phase']), ml[0], ml[1]).shape # Fetch the shape of the data
    n_ifgs_summed = np.zeros(shape) 
    # print (n_ixs[0], n_ixs[-1])
    # print (dates[n_ixs[0]], dates[n_ixs[-1]])
    for p_fn, s_fn in zip(n_ixs[:-1], n_ixs[1:]): 
        print (f"n, {p_fn = }, {s_fn = }", end='\r')
        p = np.asarray(h5.File(fns[p_fn])['Phase']) 
        s = np.asarray(h5.File(fns[s_fn])['Phase']) 

        n_ifgs_summed += np.angle(multilook(np.exp(1j*(s-p)), ml[0], ml[1]))

    # Create a daisy chain of m-day ifgs (6, 12, 18, etc. days)
    m_days = np.arange(0, length+1, m)
    m_ifgs_summed = np.zeros(shape)
    # print (m_ixs[0], m_ixs[-1])
    # print (dates[m_ixs[0]], dates[m_ixs[-1]])
    
    for p_fn, s_fn in zip(m_ixs[:-1], m_ixs[1:]):
        print (f"m, {p_fn = }, {s_fn = }", end='\r')
        p = np.asarray(h5.File(fns[p_fn])['Phase'])
        s = np.asarray(h5.File(fns[s_fn])['Phase'])

        m_ifgs_summed += np.angle(multilook(np.exp(1j*(s-p)), ml[0], ml[1]))
    
    # Find the difference between them
    diff = np.angle(np.exp(1j*(n_ifgs_summed - m_ifgs_summed)))

    # Return the difference
    return diff

def multilook_median(im, fa=3, fr=12):
    """ Averages image over multiple pixels where NaN are present and want to be maintained
    Adapted from RapidSAR's multilook func.
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
    return np.median(imout, axis=0)

def convert_landcover(im):
    out = im.copy()

    out[np.logical_and(im >= 111, im <= 126)] = 111
    out[np.logical_or(im == 90, im == 80)] = 200
    out[np.logical_or(im == 70, im == 200)] = 200
    
    return out

def landcover(arr, fp, ml):
    """
    Args:
        arr (3d array): 3d array of the loop closure. Along the zeroth axis, each 2d array is of 
                        a different m-day interferogram loop. 
    """
    types = {111:"Forest", 20: "Shrubs", 40:"Cropland", 50:"Urban", 200:"Water/Ice", 30:"Herbacious Veg.", 100:"Moss", }
    lc = h5.File(fp)["Landcover"]
    lc_ml = multilook_median(lc, ml[0], ml[1])
    lc_ml = convert_landcover(lc_ml)

    lc_loop = np.empty((arr.shape[0], 3)) # np.unique(lc_ml).shape[0]))
    for i, im in enumerate(arr):
        for l, lc_type in enumerate(np.array([111, 50, 40])): # np.unique(lc_ml)
            lc_loop[i, l] = np.angle(np.mean(np.exp(1j*im)[lc_ml == lc_type]))
    
    return lc_loop, [types[111], types[50], types[40]]

if __name__ == "__main__":
    sys.exit(main())