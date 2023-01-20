import glob
import numpy as np
import matplotlib.pyplot as plt 
import h5py as h5
from generate_a_variables import multilook
import sys

# Plot: IFG, Correction, Corrected ifg
# Plot: Loop, Loop correction, Corrected loop

def make_ifg(date1, date2):

    date1_fn = glob.glob(f"/home/jacob/Satsense/ss2/south_yorkshire/data_with_correction/IFG/singlemaster/*_{date1}/*_{date1}_ph.h5")[0]
    date2_fn = glob.glob(f"/home/jacob/Satsense/ss2/south_yorkshire/data_with_correction/IFG/singlemaster/*_{date2}/*_{date2}_ph.h5")[0]
        
    date1_ph = h5.File(date1_fn)["Phase"][:]
    date2_ph = h5.File(date2_fn)["Phase"][:]

    ifg = np.angle(multilook(np.exp(1j* (date2_ph - date1_ph) ), 3, 12))

    return ifg

def get_corr(date1, date2, a_folder):

    corr_fn = glob.glob(f"/home/jacob/Satsense/ss2/south_yorkshire/data_with_correction/2021/{a_folder}/Coherence/{date2}/{date1}-{date2}_corr.h5")[0]

    corr = h5.File(corr_fn)["Correction"][:]

    return corr

dates = [d.split("/")[-1] for d in glob.glob("/home/jacob/Satsense/ss2/south_yorkshire/data_with_correction/2021/a1*/Coherence/2021*")]
dates.sort()

np.random.seed(2021)
rand_ix = np.random.randint(3, len(dates)-4, 5)
print (rand_ix)
rand_ix = [14, 16, 18]

# IFG
# for a_folder in ["a-110_-048"]:#, , ]:
# for a_folder in ["a047_031"]:
for a_folder in ["a105_068"]:
    for i in rand_ix:

        fig_ifg, ax_ifg = plt.subplots(nrows=3, ncols=3, sharex=True, sharey=True)
        
        
        ifg12 = make_ifg(dates[i], dates[i+1])
        ifg23 = make_ifg(dates[i+1], dates[i+2])
        ifg13 = make_ifg(dates[i], dates[i+2])

        corr12 = get_corr(dates[i], dates[i+1], a_folder)
        corr23 = get_corr(dates[i+1], dates[i+2], a_folder)
        corr13 = get_corr(dates[i], dates[i+2], a_folder)

        ifg_corr12 = np.angle(np.exp(1j* (ifg12 - corr12)))
        ifg_corr23 = np.angle(np.exp(1j* (ifg23 - corr23)))
        ifg_corr13 = np.angle(np.exp(1j* (ifg13 - corr13)))
        
        for a, m in zip(ax_ifg[0, :], [ifg12, ifg23, ifg13]):
            p = a.matshow(m, vmin=-np.pi, vmax=np.pi, cmap="RdYlBu")
            plt.colorbar(p, ax=a)

        for a, m in zip(ax_ifg[1, :], [corr12, corr23, corr13]):
            p = a.matshow(m, cmap="RdYlBu")
            plt.colorbar(p, ax=a)

        for a, m in zip(ax_ifg[2, :], [ifg_corr12, ifg_corr23, ifg_corr13]):
            p = a.matshow(m, vmin=-np.pi, vmax=np.pi, cmap="RdYlBu")
            plt.colorbar(p, ax=a)

        # fig_ifg.suptitle(f"{dates[i]}-{dates[i+1]}, {dates[i+1]}-{dates[i+2]}, {dates[i]}-{dates[i+2]}")
        
        ax_ifg[0, 0].set_title(f"{dates[i]}-{dates[i+1]}", fontsize=10)
        ax_ifg[0, 1].set_title(f"{dates[i+1]}-{dates[i+2]}", fontsize=10)
        ax_ifg[0, 2].set_title(f"{dates[i]}-{dates[i+2]}", fontsize=10)

        ax_ifg[0, 0].set_ylabel("IFG")
        ax_ifg[1, 0].set_ylabel("Correction")
        ax_ifg[2, 0].set_ylabel("Corrected IFG")

        plt.savefig(f"a_comparison/{a_folder}_{dates[i]}-{dates[i+2]}_IFG.png", dpi=500)

        # fig_ifg.clear(True)
        fig_loop, ax_loop = plt.subplots(nrows=2, ncols=3)

        ifg_loop = np.angle(np.exp(1j* (ifg13 - (ifg12 + ifg23)) ))
        corr_loop = np.angle(np.exp(1j* (corr13 - (corr12 + corr23)) ))
        corr_ifg_loop = np.angle(np.exp(1j* (ifg_corr13 - (ifg_corr12 + ifg_corr23)) ))
        
        p = ax_loop[0, 0].matshow(ifg_loop, cmap="RdYlBu")
        plt.colorbar(p, ax=ax_loop[0, 0])
        p = ax_loop[0, 1].matshow(corr_loop, cmap="RdYlBu")
        plt.colorbar(p, ax=ax_loop[0, 1])
        p = ax_loop[0, 2].matshow(corr_ifg_loop, cmap="RdYlBu")
        plt.colorbar(p, ax=ax_loop[0, 2])
        
        bins = np.linspace(-np.pi, np.pi, 50)
        ax_loop[1, 0].hist(ifg_loop.flatten(), bins=bins)
        ax_loop[1, 1].hist(corr_loop.flatten(), bins=bins)
        ax_loop[1, 2].hist(corr_ifg_loop.flatten(), bins=bins)

        ax_loop[1, 0].axvline(x=0)
        ax_loop[1, 1].axvline(x=0)
        ax_loop[1, 2].axvline(x=0)

        fig_loop.suptitle(f"{dates[i]}, {dates[i+1]}, {dates[i+2]}")
        ax_loop[0, 0].set_title("IFG loop", fontsize=10)
        ax_loop[0, 1].set_title("Correction loop", fontsize=10)
        ax_loop[0, 2].set_title("Corrected IFG loop", fontsize=10)

        ax_loop[1, 0].set_title(f"mean = {np.angle(np.nanmean(np.exp(1j*ifg_loop))):.2f}", fontsize=10)
        ax_loop[1, 1].set_title(f"mean = {np.angle(np.nanmean(np.exp(1j*corr_loop))):.2f}", fontsize=10)
        ax_loop[1, 2].set_title(f"mean = {np.angle(np.nanmean(np.exp(1j*corr_ifg_loop))):.2f}", fontsize=10)

        plt.savefig(f"a_comparison/{a_folder}_{dates[i]}-{dates[i+2]}_loop.png", dpi=500)