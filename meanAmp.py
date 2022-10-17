import numpy as np
import h5py as h5
import matplotlib.pyplot as plt
import glob

wdir = "/workspace/rapidsar_test_data/south_yorkshire/jacob2/"

im_path = wdir + 'RSLC/*/*.*'

im_fns = [file for file in glob.glob(im_path, recursive=True)]

im_fns.sort()

total_amp = np.zeros(np.array(h5.File(im_fns[0])["Amplitude"], dtype=float).shape, dtype=float)

for i, fn in enumerate(im_fns[:100]):
    print (f"Image no.: {i}", end='\r')

    amp = np.array(h5.File(fn)["Amplitude"], dtype=float)

    total_amp += amp

    amp = 0

mean_amp = total_amp/100 # len(im_fns)

np.save("mean_amp.npy", mean_amp)

plt.matshow(mean_amp)
plt.show()