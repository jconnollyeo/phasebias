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
    # Read in frame    
    wdir = str(args_dict["wdir"])
    frame_ID = str(args_dict["frame"])
    # Read in date (to be corrected)
    startdate = str(args_dict["startdate"])
    corrdate = str(args_dict["corrdate"])
    date = str(args_dict["corrdate"])
    print (date)
    # Read in multilook factor
    ml = np.asarray(str(args_dict["multilook"]).split(","), dtype=int)

    plot = True
    save = False

    # Check they exist and read in other interferograms needed for loop
    fp = f"{wdir}/{frame_ID}/IFG/singlemaster/*"
    data_fns = [file for file in glob.glob(fp, recursive=True)]
    data_fns.sort()
    
    master = data_fns[0].split("/")[-1].split("_")[0]
    loop_dates = [datetime.strptime(date, "%Y%m%d") + timedelta(days=6*d) for d in [0, 1, 2, 3]]
    check = []

    for date_secondary in loop_dates:
        date_secondary_str = datetime.strftime(date_secondary, "%Y%m%d")
        fp_check = f"{fp[:-2]}/{master}_{date_secondary_str}"
        if fp_check in data_fns:
            check.append(True)  
        else:
            check.append(False)
    
    date_secondary_str = datetime.strftime(loop_dates[0], "%Y%m%d")
    fp_first = f"{fp[:-2]}/{master}_{date_secondary_str}"
    start = np.where(fp_first == np.asarray(data_fns))[0][0]
    
    check = np.array(check).all()

    if not check: 
        sys.exit(f"Not a 12 day loop.\n {data_fns[start:start+4]}")
    else: pass

    a_filename = "a_variables.npy"

    if Path(a_filename).is_file():
        a1, a2 = np.load("a_variables.npy")
        print ("a1 and a2 files read.")
    else:
        print ("Creating a1 and a2 files. ")
        phi6 = daisychain(data_fns, start, n=360, m=6, ml=ml)
        phi12 = daisychain(data_fns, start, n=360, m=12, ml=ml)
        phi18 = daisychain(data_fns, start, n=360, m=18, ml=ml)
        a1 = phi12/phi6
        a2 = phi18/phi6
        np.save("a_variables.npy", np.stack((a1, a2)))
    
    shape = a1.shape

    a1 = a1.flatten()
    a2 = a2.flatten()
    # a1 = np.mean(a1)
    # a2 = np.mean(a2)

    # print (a1, a2)
    
    a1[:] = 0.47 # From literature    
    a2[:] = 0.31 # From literature

    m_filename = "mhat_.npy"
    if Path(m_filename).is_file():
        m = np.load(m_filename) # stupid hack
        print ("mhat loaded")
    else:
        print ("Getting loop closures (d)")
        # Calc misclosure between i and i+2 [0_2 - (0_1 + 1_2)]
        # Calc misclosure between i+1 and i+3 [1_3 - (1_2 + 2_3)]
        # Calc misclosure between i and i+3 [0_3 - (0_1 + 1_2 + 2_3)] 
        closure_0_2, _ = makeLoop(start, data_fns, shape=shape, short=6, long=12, ml=ml)
        closure_1_3, _ = makeLoop(start+1, data_fns, shape=shape, short=6, long=12, ml=ml)
        closure_0_3, _ = makeLoop(start, data_fns, shape=shape, short=6, long=18, ml=ml)

        # =========================== Make plot for closure phases ================================

        fig, ax = plt.subplots(nrows=2, ncols=3)
        p = ax[0, 0].matshow(np.angle(closure_0_2), vmin=-np.pi, vmax=np.pi)
        ax[0, 1].matshow(np.angle(closure_1_3), vmin=-np.pi, vmax=np.pi)
        ax[0, 2].matshow(np.angle(closure_0_3), vmin=-np.pi, vmax=np.pi)

        plt.colorbar(p, ax=ax[:])

        ax[1, 0].hist(np.angle(closure_0_2).flatten(), bins=np.linspace(-np.pi, np.pi, 30))
        ax[1, 1].hist(np.angle(closure_1_3).flatten(), bins=np.linspace(-np.pi, np.pi, 30))
        ax[1, 2].hist(np.angle(closure_0_3).flatten(), bins=np.linspace(-np.pi, np.pi, 30))

        ax[0, 0].set_title("closure_0_2")
        ax[0, 1].set_title("closure_1_3")
        ax[0, 2].set_title("closure_0_3")

        # Make array of shape (3, 1, im.shape[0], im.shape[1])
        # This is the "data" matrix for the inversion
        d = np.stack((np.angle(closure_0_2).flatten(), np.angle(closure_1_3).flatten(), np.angle(closure_0_3).flatten()))
        
        print ("Looping through pixels\n")

        # Create the matrix for the "model" parameters
        m = np.empty((d.T.shape))
        
        dates = np.array([datetime.strftime(datetime.strptime(date, "%Y%m%d") + timedelta(days=6*d), "%Y%m%d") for d in [0, 1, 2, 3]])

        # ================================== Make the mask ======================================

        coh_0_1 = multilook(h5.File(f"{wdir}/{frame_ID}/Coherence/{dates[1]}/{dates[0]}-{dates[1]}_coh.h5")["Coherence"][:], ml[0], ml[1])
        coh_1_2 = multilook(h5.File(f"{wdir}/{frame_ID}/Coherence/{dates[2]}/{dates[1]}-{dates[2]}_coh.h5")["Coherence"][:], ml[0], ml[1])
        coh_2_3 = multilook(h5.File(f"{wdir}/{frame_ID}/Coherence/{dates[3]}/{dates[2]}-{dates[3]}_coh.h5")["Coherence"][:], ml[0], ml[1])
        coh_0_3 = multilook(h5.File(f"{wdir}/{frame_ID}/Coherence/{dates[3]}/{dates[0]}-{dates[3]}_coh.h5")["Coherence"][:], ml[0], ml[1])

        mask = (coh_0_1/255 > 0.3) & (coh_1_2/255 > 0.3) & (coh_2_3/255 > 0.3) & (coh_0_3/255 > 0.3)
        plt.matshow(mask)
        plt.colorbar()
        mask = mask.flatten()
        
        mask[:] = True

        # ============================= Loop through each pixel =================================

        for p_ix, pixel in enumerate(d.T):
            if mask[p_ix]:
                print (f"Pixel {p_ix}/{d.shape[1]} ({p_ix*100/d.shape[1]:.2f}%)", end="\r")
                # Create matrix G for the inversion and fill with a1 or a2 vals
                G = np.zeros((3, 3))
                G[0, :2] = a1[p_ix] - 1
                G[1, 1:] = a1[p_ix] - 1
                G[2, :] = a2[p_ix] - 1

                # Do the inversion - is this correct??
                # mhat = np.linalg.inv(G.transpose() @ G) @ G.transpose() @ pixel
                mhat = np.linalg.lstsq(G, pixel)
                # print (mhat)
                m[p_ix] = mhat[0]
            else:
                m[p_ix] = np.array([np.nan, np.nan, np.nan])

        np.save("mhat.npy", m)
    
    # ============ Plotting the correction for the individual interferograms of a loop ============
    fig, ax = plt.subplots(nrows=2, ncols=3)
    
    delta01 = m[:, 0].reshape(shape) # Radians
    delta12 = m[:, 1].reshape(shape)
    delta02 = np.angle(np.exp(1j*(a1*(m[:, 0]+m[:, 1])))).reshape(shape)

    p = ax[0, 0].matshow(delta01, vmin=-np.pi, vmax=np.pi)
    ax[0, 1].matshow(delta12, vmin=-np.pi, vmax=np.pi)
    ax[0, 2].matshow(delta02, vmin=-np.pi, vmax=np.pi)
    
    plt.colorbar(p, ax=ax[:])
    ax[0, 0].set_title("$\delta_{i,i+1}$")
    ax[0, 1].set_title("$\delta_{i+1,i+2}$")
    ax[0, 2].set_title("$\delta_{i,i+2}$")

    ax[1, 0].hist(m[:, 0], bins=np.linspace(-np.pi, np.pi, 30))
    ax[1, 1].hist(m[:, 1], bins=np.linspace(-np.pi, np.pi, 30))
    ax[1, 2].hist(delta02.flatten(), bins=np.linspace(-np.pi, np.pi, 30))

    # ====================================================================================

    shortcorr = np.stack((delta01, delta12))
    
    loop, loop_ = makeLoop(start, data_fns, shape=shape, short=6, long=12, ml=ml) #, mask=mask.reshape(shape)) # Make the loop (6, 6, 12)

    closure_corr = np.exp(1j*(delta02 - (delta01 + delta12))) # Create the loop using the corrections

    correct_individual_ifg(loop_, np.array([delta01, delta12, delta02]))

    print (f"{type(loop[0, 0]) = }")
    print (f"{type(closure_corr[0, 0]) = }")

    # loopc = np.angle(loop * closure_corr.conjugate()) # Loop - correction and convert to radians
    
    loopc = np.angle(loop * np.conjugate(closure_corr)) # Loop - correction and convert to radians
    print (np.isnan(loopc).all())
    plt.figure()
    plt.hist(loopc.flatten())
    loop = np.angle(loop) # Get it back to radians

    if plot:
        plot_results(loop, loopc)
    else:
        pass

    if save:
        save_corrected(loop, loopc, date)
    else:
        pass

def correct_individual_ifg(ifgs, corrections):
    
    fig, ax = plt.subplots(3, 3, sharex=True, sharey=True)
    
    corrected = np.angle(np.exp(1j*(np.angle(ifgs)-corrections)))

    for mat, a in zip(np.angle(ifgs), ax[0, :]):
        a.matshow(mat, vmin=-np.pi, vmax=np.pi)
        a.set_title("Uncorrected ifgs")
    for mat, a in zip(corrections, ax[1, :]):
        a.matshow(mat, vmin=-np.pi, vmax=np.pi)
        a.set_title("Correction for each ifg")
    for mat, a in zip(corrected, ax[2, :]):
        a.matshow(mat, vmin=-np.pi, vmax=np.pi)
        a.set_title("Corrected ifgs")

    fig.suptitle("ifg_0_6, ifg_6_12, ifg_0_12")

    # plt.figure()

    p = plt.matshow(np.angle(np.exp(1j*(corrected[-1] - (corrected[0] + corrected[1])))), vmin=-np.pi, vmax=np.pi)
    plt.colorbar(p)
    plt.title("Loop of corrected interferograms")
    
    return ax 

def plot_results(loop, loopc):
    fig, ax = plt.subplots(nrows=2, ncols=3)
    
    # print (loopc.max(), loopc.min())

    p = ax[0, 0].matshow(loop, vmin=-np.pi, vmax=np.pi)
    ax[1, 0].hist(loop.flatten(), bins=np.linspace(-np.pi, np.pi, 50))
    ax[0, 0].set_title("12, 6, 6 loop")

    ax[0, 1].matshow(loopc, vmin=-np.pi, vmax=np.pi)
    ax[1, 1].hist(loopc.flatten(), bins=np.linspace(-np.pi, np.pi, 50))
    ax[0, 1].set_title("12, 6, 6, corrected loop")

    residual = np.angle(np.exp(1j*(loopc-loop)))
    
    ax[0, 2].matshow(residual, vmin=-np.pi, vmax=np.pi)
    ax[1, 2].hist(residual.flatten(), bins=np.linspace(-np.pi, np.pi, 50))
    ax[0, 2].set_title("Residual")

    plt.colorbar(p, ax=ax[:])

    plt.show()

def save_corrected(loop, loopc, date_str):
    
    np.save(f"12_6_6/corrected/{date_str}_corrected_6-6-12.npy", np.stack((loop, loopc)))

    return True

def makeLoop(ix, fns, shape, short=6, long=12, ml=[3, 12], mask=None):

    short_fns = [f"{fn}/{fn.split('/')[-1]}_ph.h5" for fn in fns[ix:ix + int(long/short) + 1]]
                # [filename for filenames in fns[start:, start + number_of_short_ifgs_in_loop + 1]]

    long_fns  = [short_fns[0], short_fns[-1]] # First and last ifgs from the short ifgs since this is a closed loop
    
    short_ifgs = np.full((len(short_fns)-1, *shape), fill_value=np.nan, dtype=np.complex64) # Initialise the array
    i = 0 # Initialise the index for putting data into the short_ifgs array

    for ifgfn1, ifgfn2 in zip(short_fns[:-1], short_fns[1:]): # zip([fn1, fn2], [fn2, fn3])

        ifg1 = np.exp(1j*h5.File(ifgfn1, "r")["Phase"][:]) # Load in the first ifg
        ifg2 = np.exp(1j*h5.File(ifgfn2, "r")["Phase"][:]) # Load in the second ifg

        ifg12 = multilook(ifg1*ifg2.conjugate(), ml[0], ml[1]) # Create an ifg between them and multilook
        short_ifgs[i] = ifg12 # Put it in the short_ifgs array
        i += 1 # Move to the next position in the short_ifgs array
        
    ifg1 = np.exp(1j*h5.File(long_fns[0], "r")["Phase"][:]) # Load the first ifg of the ifg spanning the whole time
    ifg2 = np.exp(1j*h5.File(long_fns[1], "r")["Phase"][:]) # Load in the second ifg of the ifg spanning the whole time
    long_ifg = multilook(ifg1*ifg2.conjugate(), ml[0], ml[1]) # Create the ifg between them and multilook

    closure = long_ifg*np.prod(short_ifgs, axis=0, dtype=np.complex64).conjugate() # Compute the closure phase

    if isinstance(mask, type(None)):
        pass
    else:
        closure[mask] = np.nan + 1j*np.nan
        short_ifgs[0][mask] = np.nan + 1j*np.nan
        short_ifgs[1][mask] = np.nan + 1j*np.nan
        long_ifg[mask] = np.nan + 1j*np.nan

    return closure, np.array([short_ifgs[0], short_ifgs[1], long_ifg])


# def makeCorrectedLoop(ix, fns, shortcorr, longcorr, shape, short=6, long=12, ml=[3, 12]):
#     # Something funky is happening here.....

#     short_fns = [f"{fn}/{fn.split('/')[-1]}_ph.h5" for fn in fns[ix:ix + int(long/short) + 1]]
#     long_fns  = [short_fns[0], short_fns[-1]]
    
#     short_ifgs = np.full((len(short_fns)-1, *shape), fill_value=np.nan, dtype=np.complex64)
#     i = 0
#     for ifgfn1, ifgfn2 in zip(short_fns[:-1], short_fns[1:]):
#         ifg1 = np.exp(1j*h5.File(ifgfn1, "r")["Phase"][:])
#         ifg2 = np.exp(1j*h5.File(ifgfn2, "r")["Phase"][:])

#         ifg12 = multilook(ifg1*ifg2.conjugate(), ml[0], ml[1])
#         short_ifgs[i] = ifg12
#         i+=1
        
#     ifg1 = np.exp(1j*h5.File(long_fns[0], "r")["Phase"][:])
#     ifg2 = np.exp(1j*h5.File(long_fns[1], "r")["Phase"][:])
#     long_ifg = multilook(ifg1*ifg2.conjugate(), ml[0], ml[1])

#     closure = long_ifg*np.prod(short_ifgs, axis=0, dtype=np.complex64).conjugate()
#     closure_corr = np.exp(1j*longcorr)*np.prod(np.exp(1j*shortcorr), axis=0, dtype=np.complex64)
    
#     # closure = (long_ifg*longcorr.conjugate())*np.prod(short_ifgs*np.exp(1j*shortcorr).conjugate(), axis=0, dtype=np.complex64).conjugate()

#     return np.angle(closure*closure_corr)

def finddifferences(fns):
    """
    Used to quickly find the differences between images in a timeseries. 
    Useful for finding a chain of all 6-day interferograms.

    Args:
        fns (_type_): _description_

    Returns:
        _type_: _description_
    """

    # print (fns[0])
    fns_0 = np.array([datetime.strptime(f.split("/")[-1].split("_")[-1], "%Y%m%d") for f in fns[:-1]])
    fns_1 = np.array([datetime.strptime(f.split("/")[-1].split("_")[-1], "%Y%m%d") for f in fns[1:]])

    diff = [str(d.days) for d in (fns_1 - fns_0)]

    with open("differences.txt", "w") as f:
        f.writelines(diff)

    plt.plot(fns_0, np.asarray(diff, dtype=int))
    plt.show()

    return None

def daisychain(fns, start_ix, n=360, m=6, ml=[3, 12]):
    
    start_date = datetime.strptime(fns[start_ix].split("/")[-1].split("_")[-1], "%Y%m%d")
    end_date = datetime.strftime(start_date + timedelta(days=360), "%Y%m%d")

    master = fns[0].split('/')[-1].split('_')[0]

    try:
        end_ix = np.where(f"{fns[0][:-18]}/{master}_{end_date}" == np.array(fns))[0][0]
    except IndexError:
        sys.exit("Missing data for 360 day interferogram. ")

    # print (start_ix, end_ix)
    
    # Make the 360-day ifg
    ifgn = np.angle(multilook(np.exp(1j*h5.File(glob.glob(fns[start_ix] + "/*")[0], "r")["Phase"][:])*\
             np.exp(1j*h5.File(glob.glob(fns[end_ix] + "/*")[0], "r")["Phase"][:]).conjugate(), ml[0], ml[1]))

    # Make the chain of 6-day ifgs
    
    fns_mday_chain = [glob.glob(fns[start_ix] + "/*")[0]]
    current = [datetime.strptime(fns[start_ix].split("/")[-1].split("_")[-1], "%Y%m%d")]

    print ("Checking chain")
    while datetime.strftime(current[-1], "%Y%m%d") != end_date:
        next = current[-1] + timedelta(days=m)
        next_str = datetime.strftime(next, "%Y%m%d")
        next_fn = f"{'/'.join(fns[0].split('/')[:-1])}/{master}_{next_str}/{master}_{next_str}_ph.h5"

        if len(glob.glob(next_fn)) > 0:
            fns_mday_chain.append(next_fn)
            current.append(next)
        else:
            print (next_fn)
            sys.exit("Unable to make chain")

    fns_mday_chain.sort()

    # ifg6_ = np.exp(1j*np.array([h5.File(f, 'r')["Phase"][:] for f in fns_6day_chain]))
    # ifg6 = ifg6_[:-1]*ifg6_[1:].conjugate()

    ifgm = np.zeros(ifgn.shape, dtype=float)
    for p_fn, s_fn in zip(fns_mday_chain[:-1], fns_mday_chain[1:]):
        p = np.exp(1j*h5.File(p_fn, "r")["Phase"][:])
        s = np.exp(1j*h5.File(s_fn, "r")["Phase"][:])

        ifgm += np.angle(multilook(p*s.conjugate(), ml[0], ml[1]))
    
    return ifgn - ifgm


def parse_args():
    """
    Parses command line arguments and defines help output
    """
    parser = argparse.ArgumentParser(description='Compute and plot the closure phase.')
    parser.add_argument("-w",
                        dest="wdir", 
                        type=str,
                        help='Directory containing the triplet loop closure.',
                        default='/workspace/rapidsar_test_data/south_yorkshire')
    parser.add_argument("-f",
                        type=str,
                        dest='frame',
                        default="jacob2",
                        help='Frame ID')
    parser.add_argument("-a",
                        type=str,
                        dest='startdate',
                        default="20210421",
                        help='Start date of 360 day timeseries for a1 and a2. 20201204')
    parser.add_argument("-p",
                        type=str,
                        dest='corrdate',
                        default="20210515",
                        help="Date of ifg to be corrected")
    parser.add_argument("-m",
                        type=str,
                        dest='multilook',
                        help="Multilook factor",
                        default='3,12')
    args = parser.parse_args()
    return vars(args)

if __name__ == "__main__":
    sys.exit(main())