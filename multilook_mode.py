import numpy as np
import matplotlib.pyplot as plt
from statistics import mode
from utils import multilook
import h5py as h5 
from Fig3_Yasser import convert_landcover
a = np.array([np.random.randint(10) for _ in range(10000)]).reshape((100, 100))
# a = np.arange(10000).reshape((100, 100))

ml = [3, 12]


def multilook_mode(arr, ml, preserve_res=True):
    start_indices_i = np.arange(arr.shape[0])[::ml[0]]
    start_indices_j = np.arange(arr.shape[1])[::ml[1]]
    
    out = np.zeros(arr.shape, dtype=arr.dtype)

    for i in start_indices_i:
        for j in start_indices_j:
            out[i:i+ml[0], j:j+ml[1]] = mode(arr[i:i+ml[0], j:j+ml[1]].flatten())

    if preserve_res:
        return out
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

        return out[:end_i:ml[0], :end_j:ml[1]]

# out1 = multilook_mode(a, ml, preserve_res=True)
# out2 = multilook_mode(a, ml, preserve_res=False)

# fig, ax = plt.subplots(3)

# ax[0].pcolormesh(out1[:-(out1.shape[0] % ml[0]), :-(out1.shape[1] % ml[1])], vmin=0, vmax=10)
# ax[0].set_title(out1[:-(out1.shape[0] % ml[0]), :-(out1.shape[1] % ml[1])].shape)

# # ax[0].pcolormesh(out1, vmin=0, vmax=10)
# # ax[0].set_title(out1.shape)

# ax[1].pcolormesh(out2, vmin=0, vmax=10)
# ax[1].set_title(out2.shape)

# m = multilook(a, ml[0], ml[1])

# ax[2].pcolormesh(m)
# ax[2].set_title(m.shape)

# plt.show()
fp = "/workspace/rapidsar_test_data/south_yorkshire/datacrop_20220204.h5"
types = {111:"Forest", 20: "Shrubs", 40:"Cropland", 50:"Urban", 200:"Water/Ice", 30:"Herbacious Veg.", 100:"Moss", }
lc = h5.File(fp)["Landcover"]
# plt.matshow(np.array(lc))
# plt.show()
lc_conv = convert_landcover(np.array(lc))
# plt.matshow(lc_conv)
# plt.show()
lc_ml = multilook_mode(lc_conv, ml, preserve_res=False)
print (lc_ml.shape)
plt.matshow(lc_ml)
plt.show()
# print (lc_ml.shape)

# plt.matshow(lc_ml)
# plt.show()