from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import sys
import h5py as h5
import glob
from datetime import datetime, timedelta
import argparse
# from Fig3_Yasser import doLoops
from utils import multilook

def main():
    """This is an implimentation of the empirical correction developed by Y Maghsoudi et al. 2021. 

    Returns:
        _type_: _description_
    """
    plt.rcParams['image.cmap'] = 'RdYlBu'
    
    args_dict = parse_args()
    wdir = str(args_dict["wdir"])
    frame_ID = str(args_dict["frame"])
    startdate = str(args_dict["startdate"])
    enddate = str(args_dict["enddate"])
    ml = np.asarray(str(args_dict["multilook"]).split(","), dtype=int)
    



    plot = True
    save = True

    a1, a2 = 0.47, 0.31
    
    # Find the length of the timeseries and fetch the relevant ifg files from the specified directory and frame_ID
    timeseries_length = (datetime.strptime(enddate, "%Y%m%d") - datetime.strptime(startdate, "%Y%m%d")).days
    dates_between = [datetime.strptime(startdate, "%Y%m%d") + timedelta(days=int(d)) for d in np.arange(0, timeseries_length+1, 6)]
    ifg_filenames = [glob.glob(f"{wdir}/{frame_ID}/IFG/singlemaster/*_{datetime.strftime(date, '%Y%m%d')}/*")[0] for date in dates_between]

    assert len(ifg_filenames) >= 5, f"Currently N = {len(ifg_filenames)}, N >= 5 (overdetermined). "

    loop12, loop18 = define_loops(ifg_filenames)
    
    shape = multilook(h5.File(ifg_filenames[0])["Phase"][:], ml[0], ml[1]).shape

    # ===============================================================================
    # print ("Test Start")
    # print (loop12[0])
    # long_ifg, short_ifgs = makeLoop(loop12[0], shape=shape)

    # closure = long_ifg * np.prod(short_ifgs, axis=0).conjugate()
    # print (closure)
    # fig, ax = plt.subplots(3)
    # ax[0].matshow(np.angle(closure), vmin=-np.pi, vmax=np.pi)
    # ax[1].hist(np.angle(closure).flatten(), bins=np.linspace(-np.pi, np.pi, 30))
    # ax[2].scatter(closure.real, closure.imag)
    # plt.show()

    # print ("Test End")
    # ===============================================================================

    mask = coherence_mask(dates_between, f"{wdir}/{frame_ID}", ml, shape)
    mask[:] = True

    # =================================== Create the d matrix and make the loops ===================================

    d = np.zeros((len(loop12) + len(loop18), np.product(shape)))
    print (f"{d.shape = }")

    for i, loop_fns in enumerate(loop12 + loop18):
        print (i)
        long_ifg, short_ifgs = makeLoop(loop_fns, mask=mask, shape=shape)

        # closure = long_ifg * np.prod(short_ifgs, axis=0).conjugate() # SOMETHING HAPPENING HERE
        closure = np.exp(1j* (np.angle(long_ifg) - (np.sum(np.angle(short_ifgs), axis=0)) ) )
        # print (np.angle(closure))
        # plt.matshow(np.angle(closure), vmin=-np.pi, vmax=np.pi)
        # plt.show()
        d[i] = np.angle(closure).flatten()

    print ("Printing d")
    print (d)
    print ("\n")

    # ======================================= Create and populate the G matrix =====================================
    G12 = np.zeros((len(loop12), len(ifg_filenames)-1))
    G18 = np.zeros((len(loop18), len(ifg_filenames)-1))

    for i in range(0, len(loop12)):
        G12[i, i:i+2] = a1 - 1

    for i in range(0, len(loop18)):
        G18[i, i:i+3] = a2 - 1

    G = np.concatenate((G12, G18), axis=0)
    
    # ============================================= Perform the inversion ==========================================

    mhat = np.linalg.lstsq(G, d, rcond=None)
    # mhat[0] (4, 833000)

    # ============================================ Plot and save the data ==========================================
    if save:
        np.save(f"Correction_{startdate}_{enddate}.npy", mhat[0])
    
    if plot:
        for i, loop_fns in enumerate(loop12):
            print (i)
            long_ifg, short_ifgs = makeLoop(loop_fns, mask=mask, shape=shape)
            # loop = np.angle(np.exp(1j*(np.angle(long_ifg) - np.sum(np.angle(short_ifgs), axis=0))))

            # loop = long_ifg * np.prod(short_ifgs, axis=0).conjugate()
            loop = np.exp(1j* (np.angle(long_ifg) - (np.sum(np.angle(short_ifgs), axis=0)) ) )

            corrections = np.angle(np.exp(1j * ((a1-1)*(mhat[0][i] + mhat[0][i+1])) )).reshape(shape)
            # plt.matshow(corrections)
            ax = plot_results_map(np.angle(loop), corrections, title=str(i))

        plt.show()
        # plot_results_timeseries()


def plot_results_map(loop, correction, title=None):

    fig, ax = plt.subplots(nrows=2, ncols=3)

    p = ax[0, 0].matshow(loop, vmin=-np.pi, vmax=np.pi)
    ax[0, 1].matshow(np.angle(np.exp(1j*(loop-correction))), vmin=-np.pi, vmax=np.pi)
    ax[0, 2].matshow(correction, vmin=-np.pi, vmax=np.pi)
    
    ax[1, 0].hist(loop.flatten(), bins=np.linspace(-np.pi, np.pi, 30))
    ax[1, 1].hist(np.angle(np.exp(1j*(loop-correction))).flatten(), bins=np.linspace(-np.pi, np.pi, 30))
    ax[1, 2].hist(correction.flatten(), bins=np.linspace(-np.pi, np.pi, 30))
    
    plt.colorbar(p, ax=ax[:])
    if isinstance(title, type(None)):
        plt.suptitle("plot_results_map")
    else:
        plt.suptitle(title)
    return ax
    

def coherence_mask(dates_between, dir, ml, shape):
    
    mask = np.ones(shape, dtype=bool)

    for date1, date2 in zip(dates_between[:-1], dates_between[1:]):
        date1_str = datetime.strftime(date1, "%Y%m%d")
        date2_str = datetime.strftime(date2, "%Y%m%d")

        coherence = multilook(h5.File(f"{dir}/Coherence/{date2_str}/{date1_str}-{date2_str}_coh.h5")["Coherence"][:]/255, ml[0], ml[1])

        mask += coherence > 0.3
    
    return mask

def define_loops(ifg_filenames):

    # =================== Making 12, 6, 6 day loops ======================
    n_loops = len(ifg_filenames) - 2
    loop12 = [ifg_filenames[i:i+3] for i in range(0, n_loops)]

    # ================== Making 18, 6, 6, 6 day loops ====================
    n_loops = len(ifg_filenames) - 3
    loop18 = [ifg_filenames[i:i+4] for i in range(0, n_loops)]

    return loop12, loop18

def parse_args():
    """
    Parses command line arguments and defines help output
    """
    parser = argparse.ArgumentParser(description='Compute and plot the closure phase.')
    parser.add_argument("-w",
                        dest="wdir", 
                        type=Path,
                        help='Directory containing the triplet loop closure.',
                        default='/workspace/rapidsar_test_data/south_yorkshire')
    parser.add_argument("-f",
                        type=Path,
                        dest='frame',
                        default="jacob2",
                        help='Frame ID')
    parser.add_argument("-a",
                        type=str,
                        dest='startdate',
                        default="20210515",
                        help='Start date of 360 day timeseries for a1 and a2. 20201204')
    parser.add_argument("-p",
                        type=str,
                        dest='enddate',
                        default="20210608",
                        help="Date of ifg to be corrected")
    parser.add_argument("-m",
                        type=str,
                        dest='multilook',
                        help="Multilook factor",
                        default='3,12')
    args = parser.parse_args()
    return vars(args)

def makeLoop(filenames, shape, ml=[3, 12], mask=None):
    """Code to generate a closed loop between N interferograms. 

    Args:
        filenames (list, str): List of filenames to ifgs to construct the phase loop. 
        ml (list, optional): Multilook factor. Defaults to [3, 12].
        mask (_type_, optional): 2d array used as mask. Defaults to None.

    Returns:
        long_ifg (np.array): 2d array of ifg spanning whole time step.
        short_ifgs (np.array): 3d array of ifgs spanning 6 day steps.
    """

    short_ifgs = np.zeros((3, *shape), dtype=np.complex64)

    i=0

    for ifgfn1, ifgfn2 in zip(filenames[:-1], filenames[1:]):

        ifg1 = np.exp(1j*h5.File(ifgfn1, "r")["Phase"][:]) # Load in the first ifg
        ifg2 = np.exp(1j*h5.File(ifgfn2, "r")["Phase"][:]) # Load in the second ifg

        ifg12 = multilook(ifg1*ifg2.conjugate(), ml[0], ml[1]) # Create an ifg between them and multilook
        
        short_ifgs[i] = ifg12 # Put it in the short_ifgs array

        i += 1 # Move to the next position in the short_ifgs array

    ifg1 = np.exp(1j*h5.File(filenames[0], "r")["Phase"][:]) # Load the first ifg of the ifg spanning the whole time
    ifg2 = np.exp(1j*h5.File(filenames[-1], "r")["Phase"][:]) # Load in the second ifg of the ifg spanning the whole time
    long_ifg = multilook(ifg1*ifg2.conjugate(), ml[0], ml[1]).astype(np.complex64) # Create the ifg between them and multilook

    if isinstance(mask, type(None)):
        pass
    else:
        # closure[mask] *= np.nan
        short_ifgs[0][~mask] *= np.nan
        short_ifgs[1][~mask] *= np.nan
        long_ifg[~mask] *= np.nan

    return long_ifg, short_ifgs

if __name__ == "__main__":
    sys.exit(main())