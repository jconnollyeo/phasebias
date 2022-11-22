import sys
import numpy as np
# import matplotlib.pyplot as plt
import h5py as h5
import glob
import argparse
# from datetime import datetime

def main():
    
    args_dict = parse_args()

    wdir = str(args_dict["wdir"])
    start = int(args_dict["start"])
    length = int(args_dict["length"])
    
    dirs = glob.glob(wdir)
    dirs.sort()
    dirs = dirs[start:start+length]
    
    dates_dirs = [f.split('/')[-1] for f in dirs]

    filenames = [val for sublist in [glob.glob(dir+'/*_coh.h5') for dir in dirs] for val in sublist]

    dates_str = [d.split("-")[-1].split("_")[0] for d in filenames]
    # dates_dt = [datetime.strptime(d) for d in list(set(dates_str))]

    shape = np.asarray(h5.File(filenames[0])["Coherence"]).shape
    out = np.zeros(shape)
    count = 0
    for fn in filenames:
        with h5.File(fn) as f:
            data = np.asarray(f["Coherence"])
            out += data
            count += 1
            print (count, '\r')

    m = out/(count * 255)

    np.save(f"mean_coherence_{dates_dirs[0]}_{dates_dirs[-1]}.npy", m)


def parse_args():
    """
    Parses command line arguments and defines help output
    """
    parser = argparse.ArgumentParser(description='Compute and plot the closure phase.')
    parser.add_argument("-w",
                        dest="wdir", 
                        type=str,
                        help='Directory containing the triplet loop closure.',
                        default='/workspace/rapidsar_test_data/south_yorkshire/jacob2/Coherence/*')
    parser.add_argument("-i",
                        type=int,
                        dest='start',
                        default=286,
                        help='Start index of the loop')
    parser.add_argument("-l",
                        type=int,
                        dest='length',
                        default=60,
                        help='Length of the loop (days)')
    args = parser.parse_args()
    return vars(args)

if __name__ == "__main__":
    sys.exit(main())