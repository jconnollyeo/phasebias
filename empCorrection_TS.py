from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import sys
import h5py as h5
import glob
from datetime import datetime, timedelta
import argparse
# from utils import multilook
from generate_a_variables import multilook

def main():
    """This is an implementation of the empirical correction developed by Y Maghsoudi et al. 2021. 

    Returns:
        _type_: _description_
    """
    plt.rcParams['image.cmap'] = 'RdYlBu'
    
    args_dict = parse_args()
    wdir = str(args_dict["wdir"])
    frame_ID = str(args_dict["frame"])
    startdate, enddate = str(args_dict["daterange"]).split(",")
    ml = np.asarray(str(args_dict["multilook"]).split(","), dtype=int)
    
    plot = int(args_dict["plot"])
    save = int(args_dict["save"])

    a1, a2 = 0.47, 0.31
    # a1, a2 = 1.05, 0.68
    # a1, a2 = -1.10, -0.48

    # a1, a2 = np.load("a_variables.npy")
    
    # a1 = np.nanmean(a1)
    # a2 = np.nanmean(a2)

    print (f"{startdate = }")
    print (f"{enddate = }")

    print ("Doing timeseries checks\n")
    # Find the length of the timeseries and fetch the relevant ifg files from the specified directory and frame_ID
    timeseries_length = (datetime.strptime(enddate, "%Y%m%d") - datetime.strptime(startdate, "%Y%m%d")).days
    # Makes a list of datetime dates spanning the interval seperated by 6 days
    dates_between = [datetime.strptime(startdate, "%Y%m%d") + timedelta(days=int(d)) for d in np.arange(0, timeseries_length+1, 6)] 
    
    # Loop through each of the dates and extract the filename from the specified wdir and frame_ID
    ifg_filenames = []
    missing_files = []
    missing_files_ix = []

    for i, date in enumerate(dates_between):
        fn = glob.glob(f"{wdir}/{frame_ID}/IFG/singlemaster/*_{datetime.strftime(date, '%Y%m%d')}/*")
        if len(fn) == 1:
            ifg_filenames.append(fn[0])
        else:
            ifg_filenames.append("")
            missing_files.append(f"{wdir}/{frame_ID}/IFG/singlemaster/*_{datetime.strftime(date, '%Y%m%d')}/")
            missing_files_ix.append(i)

    # ifg_filenames = [glob.glob(f"{wdir}/{frame_ID}/IFG/singlemaster/*_{datetime.strftime(date, '%Y%m%d')}/*")[0] for date in dates_between]

    # Make sure that the number of ifg inputs is greater than 5 for an overdetermined inverse problem. 
    assert len(ifg_filenames) >= 5, f"Currently N = {len(ifg_filenames)}, N >= 5 (overdetermined). "
    
    # Make a list of lists where each sublist contains the filenames of a 12 day loop (for loop12) or 18 day loop (for loop18)
    loop12, loop18 = define_loops(ifg_filenames)
    
    # Fetch the shape of the multilooked phase - this is needed for creating output arrays going forward
    shape = multilook(h5.File(ifg_filenames[0])["Phase"][:], ml[0], ml[1]).shape
    
    # Create mask
    # av_coherence = h5.File(f"{wdir}/{frame_ID}/ruralvelcrop_20221013.h5")["Average_Coh"][:]
    av_coherence = np.ones(shape)
    # mask = av_coherence > 0.3
    mask = av_coherence > -999
    # mask = np.ones(shape, dtype=bool) # Temporary

    # =================================== Create the d matrix and make the loops ===================================
    
    # Initialise the d matrix (input)
    d = np.zeros((len(loop12) + len(loop18), np.product(shape)))
    print ("Creating d matrix")
    
    # Create the phase closure for each loop in loop12 and loop18 and assign to d
    for i, loop_fns in enumerate(loop12 + loop18):
        print (i, end="\r")
        if (np.array([len(fn) for fn in loop_fns]) == 0).any():
            d[i] = 0
        else:
            long_ifg, short_ifgs = makeLoop(loop_fns, shape=shape, ml=ml)
            closure = np.exp(1j* (np.angle(long_ifg) - (np.sum(np.angle(short_ifgs), axis=0)) ) )
            d[i] = np.angle(closure).flatten()

    print (f"{np.nanmax(d) = }, {np.nanmin(d) = }")
    print ("Finished d\n")
    # ======================================= Create and populate the G matrix =====================================
    print ("Creating G matrix")

    # Create G for the 12 day loops and 18 day loops seperately then concatenate them together.

    G12 = np.zeros((len(loop12), len(ifg_filenames)-1))
    G18 = np.zeros((len(loop18), len(ifg_filenames)-1))

    for i in range(0, len(loop12)):
        if not (np.array([len(fn) for fn in loop12[i]]) == 0).any():
            G12[i, i:i+2] = a1 - 1
        else:
            pass

    for i in range(0, len(loop18)):
        if not (np.array([len(fn) for fn in loop18[i]]) == 0).any():
            G18[i, i:i+3] = a2 - 1
        else:
            pass

    G = np.concatenate((G12, G18), axis=0)
    print ("Finished G")
    # ============================================= Perform the inversion ==========================================
    print ("Doing inversion")

    mhat = np.linalg.lstsq(G, d, rcond=None)

    print ("Finished inversion.")
    # ============================================ Plot and save the data ==========================================

    mhat[0][:, ~mask.flatten()] = np.nan # Make the low coherence pixels correction to be nan
    print (mhat)
    missing_output_ix = np.concatenate((np.array(missing_files_ix), np.array(missing_files_ix)-1))
    print (f"{missing_output_ix = }")  
    if len(missing_output_ix) > 0:
        mhat[0][missing_output_ix.astype(int), :] = np.nan
    else: pass
    print (f"{missing_output_ix = }")  

    if save:
        print ("Saving output")
        dates = [datetime.strftime(d, "%Y%m%d") for d in dates_between] 

        for i in np.arange(mhat[0].shape[0]):
            # SAVE THE 6DAY CORRECTION
            if i not in missing_output_ix:
                with h5.File(f"{wdir}/{frame_ID}/Coherence/{dates[i+1]}/{dates[i]}-{dates[i+1]}_corr.h5", "w") as f:
                    f.create_dataset("Correction", data=(mhat[0][i]).reshape(shape))
            # SAVE THE 12DAY CORRECTION
            if (i not in missing_output_ix) and (i+1 not in missing_output_ix) and (i+1 <= mhat[0].shape[0]-1):
                with h5.File(f"{wdir}/{frame_ID}/Coherence/{dates[i+2]}/{dates[i]}-{dates[i+2]}_corr.h5", "w") as f:
                    f.create_dataset("Correction", data=((a1)*(mhat[0][i] + mhat[0][i+1])).reshape(shape))
            # SAVE THE 18DAY CORRECTION
            if (i not in missing_output_ix) and (i+1 not in missing_output_ix) and (i+2 not in missing_output_ix) and (i+2 <= mhat[0].shape[0]-1):
                with h5.File(f"{wdir}/{frame_ID}/Coherence/{dates[i+3]}/{dates[i]}-{dates[i+3]}_corr.h5", "w") as f:
                    f.create_dataset("Correction", data=((a2)*(mhat[0][i] + mhat[0][i+1] + mhat[0][i+2])).reshape(shape))

        print ("Output saved")
        
    else: pass
    
    if plot:
        # If plot is true, create and plot a figure for each 12 day loop using the func "plot_results_map"
        for i, loop_fns in enumerate(loop12):

            long_ifg, short_ifgs = makeLoop(loop_fns, mask=mask, shape=shape, ml=ml)

            loop = np.exp(1j* (np.angle(long_ifg) - (np.sum(np.angle(short_ifgs), axis=0)) ) )

            corrections = np.angle(np.exp(1j * ((a1-1)*(mhat[0][i] + mhat[0][i+1])) )).reshape(shape)
            
            title = ",".join([x.split("/")[-1].split("_")[1] for x in loop_fns])

            ax = plot_results_map(np.angle(loop), corrections, title=title)

        plt.show()
    else: pass

def saveh5(fns, arr):
    for fn, m in zip(fns, arr):
        try:
            with h5.File(fn, "w") as f:
                f.create_dataset("Correction", data=m)
        except FileNotFoundError:
            print (f"Missing: {fn}")

def plot_results_map(loop, correction, title=None):
    """Plot the results in radar coords

    Args:
        loop (2d np.array): Loop closure in radians
        correction (2d np.array): Correction in radians
        title (str, optional): suptitle for the figure. Defaults to None.

    Returns:
        ax (np.array of matplotlib subplot objects): Array of subplots.
    """
    fig, ax = plt.subplots(nrows=2, ncols=3)

    p = ax[0, 0].matshow(loop, vmin=-np.pi, vmax=np.pi)
    ax[0, 1].matshow(np.angle(np.exp(1j*(loop-correction))), vmin=-np.pi, vmax=np.pi)
    ax[0, 2].matshow(correction, vmin=-np.pi, vmax=np.pi)
    
    h1, _, _ = ax[1, 0].hist(loop.flatten(), bins=np.linspace(-np.pi, np.pi, 30))
    h2, _, _ = ax[1, 1].hist(np.angle(np.exp(1j*(loop-correction))).flatten(), bins=np.linspace(-np.pi, np.pi, 30))
    h3, _, _ = ax[1, 2].hist(correction.flatten(), bins=np.linspace(-np.pi, np.pi, 30))
    
    ax[1, 0].set_title(f"Mean = {np.angle(np.mean(np.exp(1j*loop))):.2f}", horizontalalignment="center")
    ax[1, 1].set_title(f"Mean = {np.angle(np.mean(np.exp(1j*(loop-correction)))):.2f}", horizontalalignment="center")
    ax[1, 2].set_title(f"Mean = {np.angle(np.mean(np.exp(1j*correction))):.2f}", horizontalalignment="center")

    plt.colorbar(p, ax=ax[:])
    if isinstance(title, type(None)):
        plt.suptitle("plot_results_map")
    else:
        plt.suptitle(title)
    return ax

def coherence_mask(dates_between, dir, ml, shape):
    # IGNORE - DELETE
    mask = np.ones(shape, dtype=bool)

    for date1, date2 in zip(dates_between[:-1], dates_between[1:]):
        date1_str = datetime.strftime(date1, "%Y%m%d")
        date2_str = datetime.strftime(date2, "%Y%m%d")

        coherence = multilook(h5.File(f"{dir}/Coherence/{date2_str}/{date1_str}-{date2_str}_coh.h5")["Coherence"][:]/255, ml[0], ml[1])

        mask += coherence > 0.3
    
    return mask

def define_loops(ifg_filenames):
    """From the input, create a list of lists of 12 day loops and another of 18 day loops. 

    Args:
        ifg_filenames (list): List of input filenames. 

    Returns:
        loop12 (list): List of list of 12-day loops
        loop18 (list): List of list of 18-day loops
    """

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
    parser.add_argument("-d",
                        type=str,
                        dest='daterange',
                        default="20210515,20210608",
                        help="Date range of ifgs to be corrected")
    parser.add_argument("-m",
                        type=str,
                        dest='multilook',
                        help="Multilook factor",
                        default='3,12')
    parser.add_argument("-p",
                        type=int,
                        dest='plot',
                        help="Plot?",
                        default=0)
    parser.add_argument("-s",
                        type=int,
                        dest='save',
                        help="Save?",
                        default=1)              
    args = parser.parse_args()
    return vars(args)

def makeLoop(filenames, shape, ml=[3, 12], mask=None):
    """Generate a closed loop between N interferograms. 

    Args:
        filenames (list, str): List of filenames to ifgs to construct the phase loop. 
        ml (list, optional): Multilook factor. Defaults to [3, 12].
        mask (2d np.array, optional): 2d array used as mask. Defaults to None.

    Returns:
        long_ifg (np.array): 2d array of ifg spanning whole time step.
        short_ifgs (np.array): 3d array of ifgs spanning 6 day steps.
    """

    short_ifgs = np.zeros((3, *shape), dtype=np.complex64)

    i=0

    for ifgfn1, ifgfn2 in zip(filenames[:-1], filenames[1:]):

        ifg1 = np.exp(1j*h5.File(ifgfn1, "r")["Phase"][:]) # Load in the first ifg
        ifg2 = np.exp(1j*h5.File(ifgfn2, "r")["Phase"][:]) # Load in the second ifg

        # ifg12 = multilook(ifg1*ifg2.conjugate(), ml[0], ml[1]) # Create an ifg between them and multilook
        ifg12 = multilook(np.conjugate(ifg1)*ifg2, ml[0], ml[1]) # Create an ifg between them and multilook

        
        short_ifgs[i] = ifg12 # Put it in the short_ifgs array

        i += 1 # Move to the next position in the short_ifgs array

    ifg1 = np.exp(1j*h5.File(filenames[0], "r")["Phase"][:]) # Load the first ifg of the ifg spanning the whole time
    ifg2 = np.exp(1j*h5.File(filenames[-1], "r")["Phase"][:]) # Load in the second ifg of the ifg spanning the whole time
    long_ifg = multilook(np.conjugate(ifg1)*ifg2, ml[0], ml[1]).astype(np.complex64) # Create the ifg between them and multilook
    
    # Deal with mask
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