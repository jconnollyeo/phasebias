import glob
import numpy as np
import matplotlib.pyplot as plt 
import h5py as h5
from generate_a_variables import multilook
import sys
from datetime import datetime

def make_ifg(date1, date2, bounds):
    # print (date1, date2)

    date1_fn = glob.glob(f"/home/jacob/Satsense/ss2/south_yorkshire/data_with_correction/IFG/singlemaster/*_{date1}/*_{date1}_ph.h5")[0]
    date2_fn = glob.glob(f"/home/jacob/Satsense/ss2/south_yorkshire/data_with_correction/IFG/singlemaster/*_{date2}/*_{date2}_ph.h5")[0]
        
    date1_ph = h5.File(date1_fn)["Phase"][:]
    date2_ph = h5.File(date2_fn)["Phase"][:]

    ifg = np.angle(multilook(np.exp(1j* (date2_ph - date1_ph) ), 3, 12))[bounds[0]:bounds[1], bounds[2]:bounds[3]]

    return ifg

def get_corr(date1, date2, a_folder, bounds):

    corr_fn = glob.glob(f"/home/jacob/Satsense/ss2/south_yorkshire/data_with_correction/2021/{a_folder}/Coherence/{date2}/{date1}-{date2}_corr.h5")[0]

    corr = h5.File(corr_fn)["Correction"][bounds[0]:bounds[1], bounds[2]:bounds[3]]

    return corr

dates = [d.split("/")[-1] for d in glob.glob("/home/jacob/Satsense/ss2/south_yorkshire/data_with_correction/2021/a1*/Coherence/2021*")]
dates.sort()

# For each date of the time-series calculate:
# - Phase closure of a region
# - Corrected phase closure of a region

a_folder = "a105_068"

mask = np.zeros((1000, 833), dtype=bool)
mask[450:620, 680:830] = True

bounds = [450, 620, 680, 830]

ifg_region = []
corr_region = []
corr_ifg_region = []

# for i, date in zip(np.arange(4, len(dates))[:-3:2], dates[:-3:2]):

for i in np.arange(4, len(dates))[:-3:2]:

    print (f"{i/2}/{(len(dates)-4)/2}")

    ifg12 = make_ifg(dates[i], dates[i+1], bounds)
    ifg23 = make_ifg(dates[i+1], dates[i+2], bounds)
    ifg13 = make_ifg(dates[i], dates[i+2], bounds)

    corr12 = get_corr(dates[i], dates[i+1], a_folder, bounds)
    corr23 = get_corr(dates[i+1], dates[i+2], a_folder, bounds)
    corr13 = get_corr(dates[i], dates[i+2], a_folder, bounds)

    ifg_corr12 = np.angle(np.exp(1j* (ifg12 - corr12)))
    ifg_corr23 = np.angle(np.exp(1j* (ifg23 - corr23)))
    ifg_corr13 = np.angle(np.exp(1j* (ifg13 - corr13)))

    ifg_loop = np.angle(np.exp(1j* (ifg13 - (ifg12 + ifg23)) ))
    corr_loop = np.angle(np.exp(1j* (corr13 - (corr12 + corr23)) ))
    corr_ifg_loop = np.angle(np.exp(1j* (ifg_corr13 - (ifg_corr12 + ifg_corr23)) ))

    ifg_region.append(np.angle(np.mean(np.exp(1j*(ifg_loop)))))
    corr_region.append(np.angle(np.mean(np.exp(1j*(corr_loop)))))
    corr_ifg_region.append(np.angle(np.mean(np.exp(1j*(corr_ifg_loop)))))

print (np.cumsum(ifg_region))
print (np.cumsum(corr_region))
print (np.cumsum(corr_ifg_region))

dates_dt = np.array([datetime.strptime(d, "%Y%m%d") for d in dates[4:-3:2]])

plt.plot(dates_dt, -1*np.cumsum(ifg_region), label="IFG")
plt.plot(dates_dt, -1*np.cumsum(corr_region), label="Correction")
plt.plot(dates_dt, -1*np.cumsum(corr_ifg_region), label="Corrected IFG")

plt.xlabel("Date")
plt.ylabel("Loop closure")
plt.title(a_folder)
plt.xticks(rotation=45)

plt.legend()

plt.show()
