import matplotlib.pyplot as plt
import numpy as np
import glob
from datetime import datetime, timedelta
import argparse
from utils import multilook
from pathlib import Path
import sys
import h5py as h5
from loopClosure_post import convert_landcover, multilook_mode

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
    parser.add_argument("-a",
                        type=str,
                        dest='startdate',
                        default="20210103",
                        help='Start date of 360 day timeseries for a1 and a2. 20201204')
    parser.add_argument("-p",
                        type=str,
                        dest='enddate',
                        default="20211217",
                        help="Date of ifg to be corrected")
    parser.add_argument("-m",
                        type=str,
                        dest='multilook',
                        help="Multilook factor",
                        default='3,12')
    args = parser.parse_args()
    return vars(args)

def main():
    plt.rcParams['image.cmap'] = 'RdYlBu'
    
    args_dict = parse_args()
    wdir = str(args_dict["wdir"])
    frame_ID = str(args_dict["frame"])
    startdate = str(args_dict["startdate"])
    enddate = str(args_dict["enddate"])
    ml = np.asarray(str(args_dict["multilook"]).split(","), dtype=int)
    
    timeseries_length = (datetime.strptime(enddate, "%Y%m%d") - datetime.strptime(startdate, "%Y%m%d")).days
    dates_between = [datetime.strptime(startdate, "%Y%m%d") + timedelta(days=int(d)) for d in np.arange(0, timeseries_length+1, 6)]
    shape = (1000, 833)

    master_date_str = Path(glob.glob(f"{wdir}/{frame_ID}/IFG/singlemaster/*")[0]).stem.split("_")[0]
    
    d12 = np.full((len(dates_between)-1, *shape), fill_value=np.nan)

    corrections_6day = np.full((len(dates_between)-1, *shape), fill_value=np.nan)
    phases_6day = np.full((len(dates_between)-1, *shape), fill_value=np.nan)

    for i, (primary_date, secondary_date) in enumerate(zip(dates_between[:-1], dates_between[1:])):
        
        print (primary_date, secondary_date)

        primary_date_str = datetime.strftime(primary_date, "%Y%m%d")
        secondary_date_str = datetime.strftime(secondary_date, "%Y%m%d")
        try:
            correction = h5.File(f"{wdir}/{frame_ID}/Coherence/{secondary_date_str}/{primary_date_str}-{secondary_date_str}_corr.h5")["Correction"][:]
            corrections_6day[i] = correction
            ph1 = h5.File(f"{wdir}/{frame_ID}/IFG/singlemaster/{master_date_str}_{primary_date_str}/{master_date_str}_{primary_date_str}_ph.h5")["Phase"][:]
            ph2 = h5.File(f"{wdir}/{frame_ID}/IFG/singlemaster/{master_date_str}_{secondary_date_str}/{master_date_str}_{secondary_date_str}_ph.h5")["Phase"][:]
            phases_6day[i] = np.angle(multilook(np.exp(1j*(ph1 - ph2)), ml[0], ml[1]))
        except FileNotFoundError:
            pass

    corrections_12day = np.full((len(dates_between)-1, *shape), fill_value=np.nan)
    phases_12day = np.full((len(dates_between)-1, *shape), fill_value=np.nan)

    for i, (primary_date, secondary_date) in enumerate(zip(dates_between[::2], dates_between[2::2])):

        print (primary_date, secondary_date)

        primary_date_str = datetime.strftime(primary_date, "%Y%m%d")
        secondary_date_str = datetime.strftime(secondary_date, "%Y%m%d")

        try:        
            correction = h5.File(f"{wdir}/{frame_ID}/Coherence/{secondary_date_str}/{primary_date_str}-{secondary_date_str}_corr.h5")["Correction"][:]
            corrections_12day[i] = correction
            ph1 = h5.File(f"{wdir}/{frame_ID}/IFG/singlemaster/{master_date_str}_{primary_date_str}/{master_date_str}_{primary_date_str}_ph.h5")["Phase"][:]
            ph2 = h5.File(f"{wdir}/{frame_ID}/IFG/singlemaster/{master_date_str}_{secondary_date_str}/{master_date_str}_{secondary_date_str}_ph.h5")["Phase"][:]
            phases_12day[i] = np.angle(multilook(np.exp(1j*(ph1 - ph2)), ml[0], ml[1]))
        except FileNotFoundError:
            pass
    
    d12 = []
    d12_corrected = []

    for short, long, correction_short, correction_long in zip(phases_6day[::2]+phases_6day[1::2], phases_12day, corrections_6day, corrections_12day):
        closure = np.angle(np.exp(1j*(long - short)))
        closure_corrected = np.angle(np.exp(1j*((long-short) - (correction_long - correction_short))))
        d12.append(closure)
        d12_corrected.append(closure_corrected)

    plt.plot(np.angle(np.nanmean(np.exp(1j*np.array(d12)))))
    plt.plot(np.angle(np.nanmean(np.exp(1j*np.array(d12_corrected)))), label="Corrected")
    plt.show()

    sys.exit()

    # d12 = np.load(f"d_matrix_{startdate}_{enddate}_12.npy")
    # loop12 = np.load(f"d_matrix_dates_{startdate}_{enddate}_12.npy")
    # loop18 = np.load(f"d_matrix_dates_{startdate}_{enddate}_18.npy")
    # mhat = np.load(f"Correction_{startdate}_{enddate}.npy")

    mask = h5.File(f"{wdir}/{frame_ID}/ruralvelcrop_20221013.h5")["Average_Coh"][:] > 0.3
    # mask = coherence_mask(dates_between, f"{wdir}/{frame_ID}", ml, shape)
    # DO LANDCOVER

    landcover_fp = "/workspace/rapidsar_test_data/south_yorkshire/datacrop_20220204.h5"
    landcover_full = convert_landcover(h5.File(landcover_fp)["Landcover"][:])
    landcover, perc = multilook_mode(landcover_full, ml, preserve_res=False)
    shape = landcover.shape
    # END LANDCOVER

    corrections = correctionLoops(mhat)

    fig, ax = plt.subplots(nrows=2, ncols=1)

    uncorrected_mean = np.angle(np.mean(np.exp(1j*d12[::2]), axis=1))
    corrected_mean = np.angle(np.mean(np.exp(1j*(d12[::2]-corrections)), axis=1))

    uncorrected_mean_cohfilt = np.angle(np.mean(np.exp(1j*d12[::2])[:, mask.flatten()], axis=1))
    corrected_mean_cohfilt = np.angle(np.mean(np.exp(1j*(d12[::2]-corrections))[:, mask.flatten()], axis=1))

    ax[0].plot(dates_between[1::2], np.cumsum(uncorrected_mean), label="Uncorrected phase closure", color="red")
    ax[0].plot(dates_between[1::2], np.cumsum(corrected_mean), label="Corrected phase closure", color="red", ls="--")

    ax[0].plot(dates_between[1::2], np.cumsum(uncorrected_mean_cohfilt), label="Uncorrected phase closure (coh > 0.3)", color="green")
    ax[0].plot(dates_between[1::2], np.cumsum(corrected_mean_cohfilt), label="Corrected phase closure (coh > 0.3)", ls='--', color="green")

    ax[0].legend()

    ax[0].set_xlabel("Date")
    ax[0].set_ylabel("Phase (mis)closure (radians)")

    for lc_type in np.array([111, 50, 40, 20]):
        colors = {111:"olivedrab", 50:"red", 40:"hotpink", 20:"orange"}
        labels = {111:"Forest", 50:"Urban", 40:"Cropland", 20:"Shrubs (and other vegetation)"}

        lc_mask = landcover == lc_type

        uncorrected_mean_lcfilt = np.angle(np.mean(np.exp(1j*d12[::2])[:, lc_mask.flatten()], axis=1))
        corrected_mean_lcfilt = np.angle(np.mean(np.exp(1j*(d12[::2]-corrections))[:, lc_mask.flatten()], axis=1))

        ax[1].plot(dates_between[1::2], np.cumsum(uncorrected_mean_lcfilt), color=colors[lc_type], ls="-", label=labels[lc_type])
        ax[1].plot(dates_between[1::2], np.cumsum(corrected_mean_lcfilt), color=colors[lc_type], ls="--", label=labels[lc_type])
    
    ax[1].legend()

    ax[1].set_xlabel("Date")
    ax[1].set_ylabel("Phase (mis)closure (radians)")

    fig, ax = plt.subplots(nrows=2, ncols=2)

    p=ax[0, 0].matshow(np.where(mask, np.angle(np.sum(np.exp(1j*d12[::2]), axis=0)).reshape(shape), np.nan), vmin=-1, vmax=1)
    ax[0, 1].matshow(np.where(mask, np.angle(np.sum(np.exp(1j*(d12[::2]-corrections)), axis=0)).reshape(shape), np.nan), vmin=-1, vmax=1)

    ax[1, 0].hist(np.where(mask.flatten(), np.angle(np.sum(np.exp(1j*d12[::2]), axis=0)), np.nan), bins=np.linspace(-1, 1, 30))
    ax[1, 1].hist(np.where(mask.flatten(), np.angle(np.sum(np.exp(1j*(d12[::2]-corrections)), axis=0)), np.nan), bins=np.linspace(-1, 1, 30))

    cbar = plt.colorbar(p, ax=ax[:])
    cbar.ax.set_ylabel("Phase closure, radians")
    
    ax[0, 0].set_title("Cumulative sum uncorrected phase closure")
    ax[0, 1].set_title("Cumulative sum corrected phase closure")

    # uncorrected_sum, corrected_sum = sumPhaseTS(loop12, mhat, shape)
    # fig, ax = plt.subplots(nrows=1, ncols=2)

    # p = ax[0].matshow(uncorrected_sum, vmin=-50, vmax=50)
    # plt.colorbar(p, ax=ax[0])
    # p = ax[1].matshow(corrected_sum, vmin=-50, vmax=50)
    # plt.colorbar(p, ax=ax[1])
    
    # ax[0].set_title("Uncorrected wrapped phase sum")
    # ax[1].set_title("Corrected wrapped phase sum")


    plt.show()


def sumPhaseTS(loops, corrections, shape, ml=[3, 12]):
    fns = [fn for loop in loops[::2] for fn in loop[:-1]]
    
    phase = np.empty((len(fns), *shape))
    
    for i, fn in enumerate(fns):
        phase[i] = multilook(h5.File(fn)["Phase"][:], ml[0], ml[1])

    phase_6day = np.angle(np.exp(1j*(phase[:-1] - phase[1:])))

    uncorrected_sum = np.sum(phase_6day, axis=0)
    corrected_sum = np.sum(np.angle(np.exp(1j*(phase_6day-corrections.reshape(corrections.shape[0], *shape)[:-1]))), axis=0)

    return uncorrected_sum, corrected_sum
    
def correctionLoops(m):
    out = np.zeros_like(m[::2])#[:-1, :]
    a1, a2 = 0.47, 0.31
    for i, x in enumerate(range(m.shape[0] - 1)[::2]):
        out[i] = np.angle(np.exp(1j * ((a1-1)*(m[x] + m[x+1])) ))
    return out

if __name__ == "__main__":
    sys.exit(main())