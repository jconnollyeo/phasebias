import numpy as np
import h5py as h5
import matplotlib.pyplot as plt
import glob
from datetime import datetime, timedelta
import sys
from generate_a_variables import multilook
import argparse
from pathlib import Path

def main():
    args_dict = parse_args()
    wdir = args_dict["wdir"]
    frame_ID = args_dict["frame"]
    startdate, enddate = str(args_dict["daterange"]).split(",")
    ml = np.asarray(str(args_dict["multilook"]).split(","), dtype=int)
    bias = args_dict["biasflag"]
    length = 12

    phase_folder = wdir / frame_ID / "IFG/singlemaster"
    coherence_folder = wdir / frame_ID / "Coherence"
    coherence_folder = "/workspace/rapidsar_test_data/south_yorkshire/jacob2/2021/Coherence"

    coherence_folder = "/home/jacob/Satsense/ss2/south_yorkshire/data_with_correction/2021/Coherence"
    phase_folder = "/home/jacob/Satsense/ss2/south_yorkshire/data_with_correction/IFG/singlemaster"

    plt.rcParams['image.cmap'] = 'RdYlBu'

    startdate_dt = datetime.strptime(startdate, "%Y%m%d")
    enddate_dt = datetime.strptime(enddate, "%Y%m%d")

    days_between = (enddate_dt - startdate_dt).days

    corrected = []
    uncorrected = []

    for d in range(0, days_between, length):
        print (d)
        date = datetime.strftime(startdate_dt + timedelta(days=d), "%Y%m%d")
        corrected_ = loopClosure(date, phase_folder, coherence_folder, ml, length, correct=True)
        corrected.append(np.angle(np.mean(corrected_)))
        uncorrected_ = loopClosure(date, phase_folder, coherence_folder, ml, length, correct=False)
        uncorrected.append(np.angle(np.mean(uncorrected_)))

    plt.figure()
    plt.plot(np.cumsum(uncorrected), label="Uncorrected")
    plt.plot(np.cumsum(corrected), label="Corrected")
    plt.legend()
    plt.show()

def loopClosure(date, phase_folder, coherence_folder, ml=[3, 12], length = 12, correct=False):
    """Func to calc the phase loop closure. 

    Args:
        date (str): First date in the loop
        phase_folder (Path): Folder containinf the phase files
        length (int, optional): length of the loop. Defaults to 12.
    """

    days = range(0, length+1, 6)

    dates = [datetime.strftime(datetime.strptime(date, "%Y%m%d")+timedelta(days=d), "%Y%m%d") for d in days]

    phase_files = [glob.glob(f"{phase_folder}/*{d}/*") for d in dates]

    for file in phase_files:
        if len(file) == 1:
            shape = h5.File(file[0])["Phase"][:].shape
        else:
            return np.full((1000, 833), fill_value=np.nan)
    
    phase = np.zeros((len(phase_files), *shape), dtype=float)
    for i, file in enumerate(phase_files):
        phase[i] = h5.File(file[0])["Phase"][:]

    short_ifgs = np.exp(1j*(phase[1:] - phase[:-1]))
    # short_ifgs = np.conjugate(np.exp(1j*phase[:-1])) * np.exp(1j*phase[1:])

    long_ifg = np.exp(1j*(phase[-1] - phase[0]))
    # long_ifg = np.conjugate(np.exp(1j*(phase[0]))) * np.exp(1j*(phase[1]))
    
    short_ifgs_ml = np.array([multilook(short, ml[0], ml[1]) for short in short_ifgs])
    long_ifg_ml = multilook(long_ifg, ml[0], ml[1])

    short_corr = np.empty_like(short_ifgs_ml, dtype=float) # short_ifgs_ml.copy()
    
    # closure = long_ifg_ml * (np.product(short_ifgs_ml, axis=0).conjugate())

    closure = np.exp(1j* (np.angle(long_ifg_ml) - np.sum(np.angle(short_ifgs_ml), axis=0)) )
    # closure = long_ifg_ml * np.conjugate(np.product(short_ifgs_ml, axis=0))

    if correct:
        short_correction_files = [Path(f"{coherence_folder}/{d_sec}/{d_pri}-{d_sec}_corr.h5") for d_pri, d_sec in zip(dates[:-1], dates[1:])]
        for i, fn in enumerate(short_correction_files):

            if fn.exists():
                # short_corr[i] = np.exp(1j*h5.File(fn)["Correction"][:])
                short_corr[i] = h5.File(fn)["Correction"][:]


        # long_corr = np.exp(1j*h5.File(f"{coherence_folder}/{dates[-1]}/{dates[0]}-{dates[-1]}_corr.h5")["Correction"][:])
        long_corr = h5.File(f"{coherence_folder}/{dates[-1]}/{dates[0]}-{dates[-1]}_corr.h5")["Correction"][:]

        
        # closure_corr = long_corr * np.conjugate(np.product(short_corr, axis=0))
        closure_corr = np.exp(1j* (long_corr - np.sum(short_corr, axis=0)))

        # closure_corr = np.exp(1j * (np.angle(long_corr) - np.sum(np.angle(short_corr), axis=0)) )
        
        # return np.exp(1j* (np.angle(closure) - np.angle(closure_corr)) )
        return np.conjugate(closure * np.conjugate(closure_corr))
    else:
        return np.conjugate(closure)
    
    # if correct:
    #     short_correction_files = [Path(f"{coherence_folder}/{d_sec}/{d_pri}-{d_sec}_corr.h5") for d_pri, d_sec in zip(dates[:-1], dates[1:])]
    #     for i, fn in enumerate(short_correction_files):
    #         if fn.exists():
    #             short_corrected[i] = short_corrected[i]*np.exp(-1j*h5.File(fn)["Correction"][:])
    #         else: Exception(f"No correction file found: {fn}")
    #     long_corrected = long_ifg_ml * np.exp(-1j*h5.File(f"{coherence_folder}/{dates[-1]}/{dates[0]}-{dates[-1]}_corr.h5")["Correction"][:])

    #     return long_corrected * (np.product(short_corrected, axis=0).conjugate())
    # else:
    #     return long_ifg_ml * (np.product(short_ifgs_ml, axis=0).conjugate())

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
                        default="jacob2/data_with_correction",
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
    parser.add_argument("-b",
                       dest='biasflag',
                       action='store_false',
                       help="Whether or not the bias is corrected",
                       default=True)     
    args = parser.parse_args()
    return vars(args)

if __name__ == "__main__":
    sys.exit(main())