import numpy as np
import matplotlib.pyplot as plt
import glob
from datetime import datetime
from utils import multilook
# from Fig3_Yasser import multilook_median 
import h5py as h5

# Import the data
fp = '/home/jacob/phase_bias/12_6_6/data/*.npy'
# ML12_48
data_fns = [file for file in glob.glob(fp, recursive=True)]

data_fns.sort()

save = False

landcover_fp = "/workspace/rapidsar_test_data/south_yorkshire/datacrop_20220204.h5"

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

def landcover(ml, segment, fp):
    types = {111:"Forest", 20: "Shrubs", 40:"Cropland", 50:"Urban", 200:"Water/Ice", 30:"Herbacious Veg.", 100:"Moss", }
    lc = h5.File(fp)["Landcover"]
    print (lc.shape)
    lc_ml = multilook_median(lc, ml[0], ml[1])
    lc_ml = convert_landcover(lc_ml)
    # i = list(np.arange(lc_ml.shape[0])[::segment])
    # i.append(None)

    # j = list(np.arange(lc_ml.shape[1])[::segment])
    # j.append(None)

    # out = np.empty(lc_ml.shape, dtype=type(lc_ml[0, 0]))

    # for i_start, i_end in zip(i[:-1], i[1:]):
    #     for j_start, j_end in zip(j[:-1], j[1:]):
    #         out[i_start:i_end, j_start:j_end] = np.mean(lc_ml[i_start:i_end, j_start:j_end])

    return lc_ml

def extractDates(fns):
    dates_primary = []
    dates_secondary = []

    for d in fns:
        fn = d.split('/')[-1]
        dates_primary.append(datetime.strptime(fn.split('_')[0], '%Y%m%d'))
        dates_secondary.append(datetime.strptime(fn.split('_')[1].split('.')[0], '%Y%m%d'))

    return dates_primary, dates_secondary

def importData(fns):
    shape = np.load(fns[0]).shape
    out = np.empty(shape=[len(fns), *shape])

    for i, fn in enumerate(fns):
        out[i] = np.load(fn)

    return out

def splitGrids(data, size=100):
    """
    Splits an array by the size variable and computes the mean of each segment. 
    Output array is of the same shape and size of the input array. 
    """
    
    i = list(np.arange(data.shape[0])[::size])
    i.append(None)

    j = list(np.arange(data.shape[1])[::size])
    j.append(None)

    out = np.empty(data.shape, dtype=type(data[0, 0]))

    for i_start, i_end in zip(i[:-1], i[1:]):
        for j_start, j_end in zip(j[:-1], j[1:]):
            out[i_start:i_end, j_start:j_end] = np.mean(data[i_start:i_end, j_start:j_end])

    return out

def splitTS(data, size=100):
    """
    Splits each 2d array by the size variable and computes the mean of each segment and then 
    for each segment return the timeseries. 
    """
    out = np.empty(data.shape, dtype=type(data.flatten()[0]))

    for i, d in enumerate(data):
        segmented = splitGrids(d, size=size)
        out[i] = segmented
    
    return out[:, ::size, ::size]

def plotSegments(data, data_complex, ix, dates_primary, size=100):
    fig, ax = plt.subplots(nrows=1, ncols=4, figsize=(16, 5))
    p0 = ax[0].matshow(multilook(np.load("mean_amp.npy"), 3, 12), cmap="binary_r", vmin=0, vmax=3)
    p1 = ax[1].matshow(data[ix], cmap='RdYlBu', vmin=-np.pi, vmax=np.pi)
    p2 = ax[2].matshow(np.angle(splitGrids(data_complex[ix], size=size)), cmap='RdYlBu', vmin=-0.5, vmax=0.5)
    n = ax[3].hist(data[ix].flatten(), bins=np.linspace(-np.pi, np.pi, 50))[0]
    ax[3].text(-0.5, np.max(n)*0.7, f"$\mu$ = {np.mean(data[ix]):.2f}", horizontalalignment='right')
    
    cbar0 = plt.colorbar(p0, ax=ax[0], orientation='horizontal')
    cbar1 = plt.colorbar(p1, ax=ax[1], orientation='horizontal')
    cbar2 = plt.colorbar(p2, ax=ax[2], orientation='horizontal')
    
    cbar0.ax.set_xlabel("Mean amplitude")
    cbar1.ax.set_xlabel("Phase closure - 12, 6, 6")
    cbar2.ax.set_xlabel("Mean phase closure segmented")
    ax[3].set_title("Histogram of phase closure")
    
    plt.suptitle(datetime.strftime(dates_primary[ix], "%d/%m/%Y")) 
    if save:
        fig.savefig(f"12_6_6/ML12_48/segmented/{datetime.strftime(dates_primary[ix], '%Y%m%d')}_segmented.png")
        
    return None

def plotTimeseries(data, data_complex, dates_primary, size=100):
    fig, ax_TS = plt.subplots(nrows=1, ncols=2, figsize=(12, 6))
    ax_TS[0].plot(dates_primary, np.cumsum(np.array([np.mean(d) for d in data])))
    ax_TS[0].set_title("Cumulative sum of mean closure phase for entire image")
    print (data.shape, data_complex.shape)
    mdata = splitTS(data_complex, size=size)
    mdata_ = np.angle(mdata.reshape((mdata.shape[0], np.prod(mdata.shape[1:]))))
    
    # Landcover dependence

    ml = [3, 12]
    lc = landcover(ml, size, landcover_fp)
    types = {111:"Forest", 20: "Shrubs", 40:"Cropland", 50:"Urban", 200:"Water/Ice", 30:"Herbacious Veg.", 100:"Moss", }
    
    # for lc_type in [111, 50, 40]:
    #     ts = []
    #     mask = lc == lc_type
    #     # print (mask.shape)
    #     for ifg in data_complex:
    #         # print (ifg.shape)
    #         ts.append(np.angle(np.sum(ifg[mask])))
    #     ax_TS[1].plot(dates_primary, np.cumsum(np.array(ts)), label=types[lc_type])
    # ax_TS[1].legend()
    # Normal way with segments

    for m in mdata_.T:
        ax_TS[1].plot(dates_primary, np.cumsum(m))
    ax_TS[1].set_title("Cumulative sum of mean closure phase of each segment")

    # Plot the season boundaries
    plt.axvline(x=datetime.strptime("20210201", "%Y%m%d"), color="red")
    plt.axvline(x=datetime.strptime("20210501", "%Y%m%d"), color="red")
    plt.axvline(x=datetime.strptime("20210801", "%Y%m%d"), color="red")
    plt.axvline(x=datetime.strptime("20211101", "%Y%m%d"), color="red")
    if save:
        fig.savefig("12_6_6/ML12_48/timeseries_mean_loop_closure.png")
    
    return fig

def main():
    size=20
    
    dates_primary, dates_secondary = extractDates(data_fns)

    data = importData(data_fns)
    data_complex = np.exp(1j*data)
    # print (data.shape)
    # Plot the timeseries of mean closure phase for the whole image
    plotTimeseries(data, data_complex, dates_primary, size=size)
    plt.show()
    for ix in np.arange(data.shape[0]):
        plotSegments(data, data_complex, ix, dates_primary, size=size)
        # plt.close()
    plt.show()

if __name__ == "__main__":
    main()