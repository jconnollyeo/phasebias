import numpy as np
import matplotlib.pyplot as plt
import glob
from datetime import datetime
from utils import multilook
# Import the data
fp = '/home/jacob/phase_bias/12_6_6/data/*.npy'

data_fns = [file for file in glob.glob(fp, recursive=True)]

data_fns.sort()

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

def main():
    size=50
    
    dates_primary, dates_secondary = extractDates(data_fns)

    data = importData(data_fns)
    data_complex = np.exp(1j*data)

    print (data.shape)

    # Plot the timeseries of mean closure phase for the whole image
    fig, ax_TS = plt.subplots(nrows=1, ncols=2, figsize=(12, 6))
    ax_TS[0].plot(dates_primary, np.array([np.mean(d) for d in data]))
    ax_TS[0].set_title("Mean closure phase for entire image")

    mdata = splitTS(data_complex, size=size)
    mdata_ = np.angle(mdata.reshape((mdata.shape[0], np.prod(mdata.shape[1:]))))

    for m in mdata_.T:
        ax_TS[1].plot(dates_primary, m)
    ax_TS[1].set_title("Mean closure phase of each segment")

    # Plot the season boundaries
    plt.axvline(x=datetime.strptime("20210201", "%Y%m%d"), color="red")
    plt.axvline(x=datetime.strptime("20210501", "%Y%m%d"), color="red")
    plt.axvline(x=datetime.strptime("20210801", "%Y%m%d"), color="red")
    plt.axvline(x=datetime.strptime("20211101", "%Y%m%d"), color="red")

    fig, ax = plt.subplots(nrows=1, ncols=4, figsize=(16, 5))
    ix = 10
    p0 = ax[0].matshow(multilook(np.load("mean_amp.npy"), 3, 12), cmap="binary_r", vmax=3, vmin=0)
    p1 = ax[1].matshow(data[ix], cmap='RdYlBu', vmin=-np.pi, vmax=np.pi)
    p2 = ax[2].matshow(np.angle(splitGrids(data_complex[ix], size=size)), cmap='RdYlBu', vmin=-0.5, vmax=0.5)
    n = ax[3].hist(data[ix].flatten(), bins=np.linspace(-np.pi, np.pi, 50))[0]
    ax[3].text(-0.5, np.max(n)*0.7, f"$\mu$ = {np.mean(data[ix]):.2f}", horizontalalignment='right')
    
    plt.colorbar(p0, ax=ax[0], orientation='horizontal')
    plt.colorbar(p1, ax=ax[1], orientation='horizontal')
    plt.colorbar(p2, ax=ax[2], orientation='horizontal')
    
    ax[0].set_title("Mean amplitude")
    ax[1].set_title("Phase closure - 12, 6, 6")
    ax[2].set_title("Mean phase closure segmented")
    ax[3].set_title("Histogram of phase closure")

    

    plt.show()

if __name__ == "__main__":
    main()