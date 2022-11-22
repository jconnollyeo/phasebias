import numpy as np
import matplotlib.pyplot as plt
import glob
from datetime import datetime, timedelta
from utils import multilook
# from Fig3_Yasser import multilook_median 
import h5py as h5
from statistics import mode
import matplotlib.dates as mdates

# Import the data
fp = '/home/jacob/phase_bias/12_6_6/data/*.npy'
# ML12_48
data_fns = [file for file in glob.glob(fp, recursive=True)]

data_fns.sort()

save = False

landcover_fp = "/workspace/rapidsar_test_data/south_yorkshire/datacrop_20220204.h5"

rain = ['140.2', '105.0', '86.8', '20.6', '121.4', '44.7', '75.3', '66.5', '82.4', '168.7', '78.1', '114.4']
rain = [float(r) for r in rain]
temp = np.array([3.3, 5.1, 7.3, 6.6, 10.1, 15.4, 17.6, 16.1, 15.9, 12.1, 7.8, 6.4])

months = np.array([datetime.strptime(str(int(data_fns[0].split("/")[-1][:6])+m), "%Y%m") for m in range(12)])

rain_bool = False

def multilook_mode(arr, ml, preserve_res=True):
    start_indices_i = np.arange(arr.shape[0])[::ml[0]]
    start_indices_j = np.arange(arr.shape[1])[::ml[1]]
    
    out = np.zeros(arr.shape, dtype=arr.dtype)

    perc = np.zeros(arr.shape, dtype=float)

    for i in start_indices_i:
        for j in start_indices_j:
            m = mode(arr[i:i+ml[0], j:j+ml[1]].flatten())
            out[i:i+ml[0], j:j+ml[1]] = m
            p = np.sum(arr[i:i+ml[0], j:j+ml[1]] == m)/\
                        (out[i:i+ml[0], j:j+ml[1]].size)
            # print (p)
            perc[i:i+ml[0], j:j+ml[1]] = p

    
    print (np.mean(perc), np.std(perc))

    if preserve_res:
        return out, perc
    else:
        end_i = out.shape[0] % ml[0]
        end_j = out.shape[1] % ml[1]

        if end_i == 0:
            end_i = None
        else:
            end_i = -1*end_i

        if end_j == 0:
            end_j = None
        else:
            end_j = -1*end_j

        out = out[:end_i:ml[0], :end_j:ml[1]]
        perc = perc[:end_i:ml[0], :end_j:ml[1]]

        return out, perc

def convert_landcover(im):
    """
    Keeps the urban,  classification the same.

    Args:
        im (_type_): _description_

    Returns:
        _type_: _description_
    """
    out = im.copy()

    out[np.logical_and(im >= 111, im <= 126)] = 111 # Forests
    out[np.logical_or(im == 90, im == 80)] = 200 # Water
    out[np.logical_or(im == 70, im == 200)] = 200 # Water
    out[np.logical_or(im == 20, im == 30)] = 20 # shrubs, herb veg, moss
    out[np.logical_or(im == 20, im == 100)] = 20 

    return out

def landcover(arr, fp, ml):
    """
    Args:
        arr (3d array): 3d array of the loop closure. Along the zeroth axis, each 2d array is of 
                        a different m-day interferogram loop. 
    """
    types = {111:"Forest", 20: "Shrubs", 40:"Cropland", 50:"Urban", 200:"Water/Ice", 30:"Herbacious Veg.", 100:"Moss", }
    lc = h5.File(fp)["Landcover"]
    lc_conv = convert_landcover(np.array(lc))
    lc_ml, perc = multilook_mode(lc_conv, ml, preserve_res=False)

    perc_mask = perc > 0.7
    print (lc_ml.shape)

    # coh_mask = multilook(np.load("mean_coherence_20201228_20211217.npy"), ml[0], ml[1]) > 0.3
    coh_mask = multilook(np.load("mean_coherence_20191228_20201228.npy"), ml[0], ml[1]) > 0.3
    
    lc_loop = np.empty((arr.shape[0], 4))
    for i, im in enumerate(arr):
        for l, lc_type in enumerate(np.array([111, 50, 40, 20])):
            lc_loop[i, l] = np.angle(np.mean(np.exp(1j*im)[(lc_ml == lc_type) & coh_mask & perc_mask]))
            # lc_loop[i, l] = np.angle(np.mean(np.exp(1j*im)[(lc_ml == lc_type)])) 

    return lc_loop, [types[111], types[50], types[40], types[20]]

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
    # plt.style.use(['seaborn-poster'])    

    fig, ax_TS = plt.subplots(nrows=1, ncols=2, figsize=(12, 6))
    ax_TS[0].plot(dates_primary, np.cumsum(np.array([np.mean(d) for d in data])))
    if rain_bool:
        ax_TS0_rain = ax_TS[0].twinx()
        ax_TS0_rain.plot(months, rain, 'b--', label="Rainfall")
        ax_TS0_rain.plot(months, temp*4, 'r--', label="Temperature")

        ax_TS0_rain.legend(loc=4)
    
    ax_TS[0].set_title("Cumulative sum of mean closure phase for entire image")
    print (data.shape, data_complex.shape)
    mdata = splitTS(data_complex, size=size)
    mdata_ = np.angle(mdata.reshape((mdata.shape[0], np.prod(mdata.shape[1:]))))
    
    # Landcover dependence
    landcover_fp = "/workspace/rapidsar_test_data/south_yorkshire/datacrop_20220204.h5"

    ml = [3, 12]
    lc, lc_types = landcover(data, landcover_fp, ml)
    types = {111:"Forest", 20: "Shrubs", 40:"Cropland", 50:"Urban", 200:"Water/Ice", 30:"Herbacious Veg.", 100:"Moss", }
    print (np.array(lc).shape)
    
    for l, lc_type in enumerate([111, 50, 40, 20]):
        ax_TS[1].plot(dates_primary, np.cumsum(np.array(lc[:, l])), label=types[lc_type])
    if rain_bool:
        ax_TS1_rain = ax_TS[1].twinx()
        ax_TS1_rain.plot(months, rain, 'b--', label="Rainfall")
        ax_TS1_rain.plot(months, temp*4, 'r--', label="Temperature")
        ax_TS1_rain.legend(loc=4)
    ax_TS[1].legend()
    
    # Normal way with segments

    ax_TS[1].set_title("Cumulative sum of mean closure phase of each landcover")

    # Plot the season boundaries
    plt.axvline(x=datetime.strptime(data_fns[0].split("/")[-1][:4] + "0201", "%Y%m%d"), color="red", alpha=0.4)
    plt.axvline(x=datetime.strptime(data_fns[0].split("/")[-1][:4] + "0501", "%Y%m%d"), color="red", alpha=0.4)
    plt.axvline(x=datetime.strptime(data_fns[0].split("/")[-1][:4] + "0801", "%Y%m%d"), color="red", alpha=0.4)
    plt.axvline(x=datetime.strptime(data_fns[0].split("/")[-1][:4] + "1101", "%Y%m%d"), color="red", alpha=0.4)
    if save:
        fig.savefig("12_6_6/ML12_48/timeseries_mean_loop_closure.png")
    dates_middle = [d+timedelta(days=6) for d in dates_primary]
    monthly = [datetime.strptime(str(202101 + m), "%Y%m") for m in range(12)]

    # For presentation
    fig, ax = plt.subplots(figsize=(12, 8))
    for l, lc_type in enumerate([111, 50, 40, 20]):
        ax.plot(dates_middle, np.cumsum(np.array(lc[:, l])), label=types[lc_type])
    ax.legend()
    ax.set_title("Cumulative sum of mean closure phase by landcover type")
    ax.set_ylabel(r"Phase closure, $\phi_{loop}$ (radians)")
    ax.set_xlabel("Date")

    ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=range(1,13,1)))
    # you can change the format of the label (now it is 2016-Jan)  
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%b'))
    plt.setp(ax.get_xticklabels(), rotation=45) 
    plt.grid()
    
    
    plt.show()
    


    return fig

def main():
    size=40
    
    dates_primary, dates_secondary = extractDates(data_fns)

    data = importData(data_fns)
    data_complex = np.exp(1j*data)
    # print (data.shape)
    # Plot the timeseries of mean closure phase for the whole image
    plotTimeseries(data, data_complex, dates_primary, size=size)
    plt.show()
    # for ix in np.arange(data.shape[0]):
    #     plotSegments(data, data_complex, ix, dates_primary, size=size)
        # plt.close()
    plt.show()

if __name__ == "__main__":
    main()