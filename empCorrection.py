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
    # Read in multilook factor
    ml = np.asarray(str(args_dict["multilook"]).split(","), dtype=int)
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
    
    # print (start)
    check = np.array(check).all()

    if not check: sys.exit(f"Not a 12 day loop.\n {data_fns[start:start+4]}")
    
    # If a1 and a2 files cannot be found:
    #     Find a1
    #     Find a2
    try:
        a1, a2 = np.load("a_variables.npy")
        print ("a1 and a2 files read.")
    except FileNotFoundError:
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
    
    a1[:] = 0.47
    a2[:] = 0.31
    
    try:
        m = np.load("mhat.npy")
        # m = np.load("asdasdamhat.npy") # stupid hack
        print ("m loaded")
    except FileNotFoundError:
        # print (a1, a2)
        
        # Form G
        # print ("Forming G")
        # G = np.zeros((3, 3))
        # G[0, :2] = a1 - 1
        # G[1, 1:] = a1 - 1
        # G[2, :] = a2 - 1
        # print (G)
        
        # Form d

        print ("Forming d")
        # Calc misclosure between i and i+2 [0_2 - (0_1 + 1_2)]
        # Calc misclosure between i+1 and i+3 [1_3 - (1_2 + 2_3)]
        # Calc misclosure between i and i+3 [0_3 - (0_1 + 1_2 + 2_3)] 
        closure_0_2 = makeLoop(start, data_fns, shape=shape, short=6, long=12, ml=ml)
        closure_1_3 = makeLoop(start+1, data_fns, shape=shape, short=6, long=12, ml=ml)
        closure_0_3 = makeLoop(start, data_fns, shape=shape, short=6, long=18, ml=ml)

        print (closure_0_2)
        print (closure_1_3)
        print (closure_0_3)

        # Make array of shape (3, 1, im.shape[0], im.shape[1])
        # How to apply a matrix mult to each value in another matrix
        d = np.stack((closure_0_2.flatten(), closure_1_3.flatten(), closure_0_3.flatten()))
        
        m = np.empty((d.T.shape))
        for p_ix, pixel in enumerate(d.T):
            
            G = np.zeros((3, 3))
            G[0, :2] = a1[p_ix] - 1
            G[1, 1:] = a1[p_ix] - 1
            G[2, :] = a2[p_ix] - 1

            print (f"Pixel {p_ix}/{d.shape[1]} ({p_ix*100/d.shape[1]:.2f}%)", "\r")
            mhat = np.linalg.inv(G.transpose() @ G) @ G.transpose() @ pixel
            m[p_ix] = mhat
        # Perform inversion
        # mhat = np.linalg.inv ( G.transpose() @ G ) @ G.transpose() @ d
        
        np.save("mhat.npy", m)
        print (m.shape)
    
    # ======= Plotting the correction for the individual interferograms of a loop ========

    fig, ax = plt.subplots(nrows=1, ncols=3, sharex=True, sharey=True)
    
    delta01 = np.angle(np.exp(1j*m[:, 0].reshape(shape)))
    delta12 = np.angle(np.exp(1j*m[:, 1].reshape(shape)))

    delta02 = np.angle(np.exp(1j*a1.reshape(shape)*(delta01+delta12)))

    p = ax[0].matshow(delta01, vmin=-np.pi, vmax=np.pi)
    ax[1].matshow(delta12, vmin=-np.pi, vmax=np.pi)
    ax[2].matshow(delta02, vmin=-np.pi, vmax=np.pi)
    
    plt.colorbar(p, ax=ax[:])
    ax[0].set_title("$\delta_{i,i+1}$")
    ax[1].set_title("$\delta_{i+1,i+2}$")
    ax[2].set_title("$\delta_{i,i+2}$")

    shortcorr = np.stack((delta01, delta12))
    
    loop = makeLoop(start, data_fns, shape=shape, short=6, long=12, ml=ml)
    loopc = makeCorrectedLoop(start, data_fns, shape=shape, shortcorr=shortcorr, longcorr=delta02, short=6, long=12, ml=ml)

    fig, ax = plt.subplots(nrows=1, ncols=3, sharex=True, sharey=True)
    print (loopc.max(), loopc.min())
    p = ax[0].matshow(loop, vmin=-np.pi, vmax=np.pi)
    # p = ax[0].matshow(np.isnan(loop), vmin=0, vmax=1)
    ax[0].set_title("12, 6, 6 loop")

    ax[1].matshow(loopc, vmin=-np.pi, vmax=np.pi)
    # ax[1].matshow(np.isnan(loopc), vmin=0, vmax=1)
    ax[1].set_title("12, 6, 6, corrected loop")

    ax[2].matshow(loopc-loop, vmin=-np.pi, vmax=np.pi)
    ax[2].set_title("Residual")

    # plt.colorbar(p, ax=ax[:])

    plt.show()

# def makeLoop(ix, fns, short=6, long=12, ml=[3,12]):

#     short_fns = [f"{fn}/{fn.split('/')[-1]}_ph.h5" for fn in fns[ix:ix + int(long/short) + 1]]
#     long_fns  = [short_fns[0], short_fns[-1]]
#     for s in short_fns:
#         print (s)

#     print ("")
#     print (long_fns)

#     short_ifgs = np.sum(np.array([np.angle(np.exp(1j*h5.File(short_fns[i])["Phase"][:])*\
#         np.exp(1j*h5.File(short_fns[i+1])["Phase"][:]).conjugate()) for i in range(len(short_fns)-1)]), axis=0)
#     long_ifgs = np.angle(np.exp(1j*h5.File(long_fns[0])["Phase"][:])*np.exp(1j*h5.File(long_fns[1])["Phase"][:]).conjugate())
    
#     return np.angle(np.exp(1j*(multilook(long_ifgs, ml[0], ml[1])-multilook(short_ifgs, ml[0], ml[1]))))

def makeLoop(ix, fns, shape, short=6, long=12, ml=[3, 12]):

    short_fns = [f"{fn}/{fn.split('/')[-1]}_ph.h5" for fn in fns[ix:ix + int(long/short) + 1]]
    long_fns  = [short_fns[0], short_fns[-1]]
    
    short_ifgs = np.full((len(short_fns)-1, *shape), fill_value=np.nan, dtype=np.complex64)
    i = 0
    for ifgfn1, ifgfn2 in zip(short_fns[:-1], short_fns[1:]):
        ifg1 = np.exp(1j*h5.File(ifgfn1, "r")["Phase"][:])
        ifg2 = np.exp(1j*h5.File(ifgfn2, "r")["Phase"][:])

        ifg12 = multilook(ifg1*ifg2.conjugate(), ml[0], ml[1])
        short_ifgs[i] = ifg12
        i += 1
        
    ifg1 = np.exp(1j*h5.File(long_fns[0], "r")["Phase"][:])
    ifg2 = np.exp(1j*h5.File(long_fns[1], "r")["Phase"][:])
    long_ifg = multilook(ifg1*ifg2.conjugate(), ml[0], ml[1])
    
    closure = long_ifg*np.prod(short_ifgs, axis=0, dtype=np.complex64).conjugate()

    return np.angle(closure)


def makeCorrectedLoop(ix, fns, shortcorr, longcorr, shape, short=6, long=12, ml=[3, 12]):

    short_fns = [f"{fn}/{fn.split('/')[-1]}_ph.h5" for fn in fns[ix:ix + int(long/short) + 1]]
    long_fns  = [short_fns[0], short_fns[-1]]
    
    short_ifgs = np.full((len(short_fns)-1, *shape), fill_value=np.nan, dtype=np.complex64)
    i = 0
    for ifgfn1, ifgfn2 in zip(short_fns[:-1], short_fns[1:]):
        ifg1 = np.exp(1j*h5.File(ifgfn1, "r")["Phase"][:])
        ifg2 = np.exp(1j*h5.File(ifgfn2, "r")["Phase"][:])

        ifg12 = multilook(ifg1*ifg2.conjugate(), ml[0], ml[1])
        short_ifgs[i] = ifg12
        
    ifg1 = np.exp(1j*h5.File(long_fns[0], "r")["Phase"][:])
    ifg2 = np.exp(1j*h5.File(long_fns[1], "r")["Phase"][:])
    long_ifg = multilook(ifg1*ifg2.conjugate(), ml[0], ml[1])

    closure = (long_ifg*longcorr.conjugate())*np.prod(short_ifgs*np.exp(1j*shortcorr).conjugate(), axis=0, dtype=np.complex64).conjugate()

    return np.angle(closure)

# def makeCorrectedLoop(ix, fns, shortcorr, longcorr, short=6, long=12, ml=[3,12]):

#     print (shortcorr.shape)
#     print (longcorr.shape)

#     short_fns = [f"{fn}/{fn.split('/')[-1]}_ph.h5" for fn in fns[ix:ix + int(long/short) + 1]]
#     long_fns  = [short_fns[0], short_fns[-1]]

#     short_ifgs = np.array([multilook(np.exp(1j*h5.File(short_fns[i])["Phase"][:])*\
#         np.exp(1j*h5.File(short_fns[i+1])["Phase"][:]).conjugate(), ml[0], ml[1]) for i in range(len(short_fns)-1)])

#     print (short_ifgs.shape)
#     short_ifgs_corr = np.sum(np.angle(short_ifgs*shortcorr.conjugate()), axis=0)

#     long_ifgs_corr = np.angle(multilook(np.exp(1j*h5.File(long_fns[0])["Phase"][:])*np.exp(1j*h5.File(long_fns[1])["Phase"][:]).conjugate(), ml[0], ml[1])*longcorr.conjugate())

#     return np.angle(np.exp(1j*(long_ifgs_corr-short_ifgs_corr)))

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
                        default="20201204",
                        help='Start date of 360 day timeseries for a1 and a2. ')
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