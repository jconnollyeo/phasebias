import numpy as np
import matplotlib.pyplot as plt
import h5py as h5

with h5.File("/workspace/rapidsar_test_data/south_yorkshire/jacob2/velcrop_20221013.h5") as f:
    reliable = np.asarray(f["Reliable_Pixels"])

plt.matshow(reliable)
plt.show()
