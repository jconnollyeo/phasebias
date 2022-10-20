import numpy as np
import matplotlib.pyplot as plt
import glob
from datetime import date, datetime

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

    

    return ""

def main():
    dates_primary, dates_secondary = extractDates(data_fns)

    data = importData(data_fns)
    print (data.shape)

    # Plot the timeseries of mean closure phase for the whole image
    plt.figure()
    plt.plot(dates_primary, np.array([np.mean(d) for d in data]))
    plt.title("Mean closure phase for entire image")

    # Plot the season boundaries
    plt.axvline(x=datetime.strptime("20210201", "%Y%m%d"), color="red")
    plt.axvline(x=datetime.strptime("20210501", "%Y%m%d"), color="red")
    plt.axvline(x=datetime.strptime("20210801", "%Y%m%d"), color="red")
    plt.axvline(x=datetime.strptime("20211101", "%Y%m%d"), color="red")


    plt.show()

if __name__ == "__main__":
    main()