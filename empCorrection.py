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

    args_dict = parse_args()
    # Read in frame    
    wdir = str(args_dict["wdir"])
    frame_ID = str(args_dict["frame"])
    # Read in date (to be corrected)
    date = str(args_dict["date"])
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
    print (start)
    check = np.array(check).all()

    if not check: sys.exit(f"Not a 12 day loop.\n {data_fns[start:start+4]}")

    # Read in filepaths for a1 and a2
    
    # If a1 and a2 files cannot be found:
    #     Find a1
    #     Find a2
    try:
        a1, a2 = np.load("a_variables.npy")
    except FileNotFoundError:
        phi6 = daisychain(data_fns, start, n=360, m=6, ml=ml)
        phi12 = daisychain(data_fns, start, n=360, m=12, ml=ml)
        phi18 = daisychain(data_fns, start, n=360, m=18, ml=ml)
        a1 = phi12/phi6
        a2 = phi18/phi6
        np.save("a_variables.npy", np.stack(a1, a2))

    # Form G
    # Form d

    # Perform inversion
    # mhat = np.linalg.inv ( G.transpose() @ G ) @ G.transpose() @ d
    
    # Correct interferogram
    # Make fig:
    #     Before, after, residual
    # Make fig:
    #     a1, a2

    # return corrected, residual
    
    return None

def checkchain(start_ix, end_ix, fns):
    dates = np.array([datetime.strptime(f.split("/")[-1].split("_")[-1], "%Y%m%d") for f in fns])



def daisychain(fns, start_ix, n=360, m=6, ml=[3, 12]):
    
    start_date = datetime.strptime(fns[start_ix].split("/")[-1].split("_")[-1], "%Y%m%d")
    end_date = datetime.strftime(start_date + timedelta(days=360), "%Y%m%d")

    master = fns[0].split('/')[-1].split('_')[0]

    end_ix = np.where(f"{fns[0][:-18]}/{master}_{end_date}" == np.array(fns))[0][0]
    print (start_ix, end_ix)
    
    # Make the 360-day ifg
    ifg360 = multilook(h5.File(glob.glob(fns[start_ix] + "/*")[0], "r")["Phase"][:]*\
             h5.File(glob.glob(fns[end_ix] + "/*")[0], "r")["Phase"][:].conjugate(), ml[0], ml[1])

    # Make the chain of 6-day ifgs
    ix_6day = []
    current = datetime.strptime("20000000", "%Y%m%d")
    np.arange(0, int(n/m), 1)
    while current != end_date:
        
        

    ifg6_1 = np.array([h5.File(glob.glob(f"{fns[ifg]}/*")) for ifg in ix_6day])
    

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
    parser.add_argument("-p",
                        type=str,
                        dest='date',
                        default="20210103",
                        help='Date')
    parser.add_argument("-m",
                        type=str,
                        dest='multilook',
                        help="Multilook factor",
                        default='3,12')
    args = parser.parse_args()
    return vars(args)

if __name__ == "__main__":
    sys.exit(main())