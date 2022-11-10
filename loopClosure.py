from ast import arg
import numpy as np
import h5py as h5
import matplotlib.pyplot as plt
# import os
import glob
from datetime import datetime
import sys
from utils import multilook
import argparse
# from pathlib import Path
import matplotlib.transforms as mtrans

# wdir = "/workspace/rapidsar_test_data/south_yorkshire/jacob2/"

def newIFGs(primary_ix, secondary_ix, im_fns):
    """
    Produce a new ifgs spanning the two indices from h5 files using index
    in wdir. 
    """
    p1 = np.exp(1j*np.asarray(h5.File(im_fns[primary_ix])['Phase']))
    p2 = np.exp(1j*np.asarray(h5.File(im_fns[secondary_ix])['Phase']))
    
    ifg = p2*p1.conjugate()

    return ifg

def create_loop(start, end, dates, im_fns, ml=[3, 12]):
    # Getting the dates from the user input indices
    d_ = dates[start:end+1]
    if np.sum(np.array(ml)) != 0:
        ml_bool = True
    else:
        ml_bool = False
    if ml_bool:
        shape = multilook(newIFGs(0, 1, im_fns), ml[0], ml[1]).shape
    else:
        shape = newIFGs(0, 1, im_fns).shape

    # Creating the interferograms 
    ifgs = np.empty((len(d_)-1, *shape), dtype=np.complex64)
    
    print ("Creating interferograms")
    for i, d in enumerate(np.arange(start, end)):
        if ml_bool:
            ifgs[i] = multilook(newIFGs(d, d+1, im_fns), ml[0], ml[1]) # Complex np.complex64
        else:
            pass

    print ("Creating interferogram spanning whole range")
    ifg_span = newIFGs(start, end, im_fns)
    if ml_bool:
        ifg_span = multilook(ifg_span, ml[0], ml[1]) # Complex np.complex64
    else: pass
    return ifgs, ifg_span

def phase_closure(ifgs, ifg_span):
    """
    Produce the phase closure of the defined indices. This will be a series 
    of ifgs of the shortest possible baseline and one spanning the entire time. 
    """

    print ("Computing the closure phase")
    closure = ifg_span*np.prod(ifgs, axis=0, dtype=np.complex64).conjugate() # Phi_start, end - (Phi_start,start+1 + Phi_start+1,start+2, etc.)

    # Returning the phase closure in radians and the figure title
    return np.angle(closure)

def plot_fig2(ifgs, ifg_span, closure):

    num_loop = ifgs.shape[0]

    fig, ax = plt.subplots(nrows=1, ncols=int(1+num_loop+1), figsize=(4*(num_loop + 2), 6))

    p = ax[0].matshow(np.angle(ifg_span), cmap="RdYlBu", vmax=np.pi, vmin=-np.pi)
    for i, a in enumerate(ax[1:-1]):
        a.matshow(np.angle(ifgs[i]), cmap="RdYlBu", vmax=np.pi, vmin=-np.pi)
        
    ax[-1].matshow(closure, cmap="RdYlBu", vmax=np.pi, vmin=-np.pi)
    # ax[0].set_title(titles[0])
    # ax[1].set_title(titles[1])
    # ax[2].set_title(titles[2])

    cbar = plt.colorbar(p, ax=ax, shrink=0.6)
    cbar.ax.set_yticks(
        [-np.pi, 0, np.pi],
        [r"$-\pi$", "0", r"$\pi$"],
    )
    cbar.ax.set_ylabel("Closure phase (radians)")
    for a in ax:
        a.axis('off')
    
    xs, ys = find_bbox(fig, ax)
    
    for l in left_bracket(xs, ys, 1):
        fig.add_artist(l)

    for l in right_bracket(xs, ys, 3):
        fig.add_artist(l)

    for index in np.arange(2, len(xs)-2):
        for l in plus(xs, index, fig):
            fig.add_artist(l)
    
    plt.show()

def left_bracket(x, y, index):

    linever = plt.Line2D([x[index], x[index]], [y[0]-0.01, y[1]+0.01], color='black')
    linetop = plt.Line2D([x[index], x[index]+0.01], [y[1]+0.01, y[1]+0.01], color='black')
    linebot = plt.Line2D([x[index], x[index]+0.01], [y[0]-0.01, y[0]-0.01], color='black')

    minus = plt.Line2D([x[index] - 0.005, x[index] - 0.009], [0.5, 0.5], color='black')
    
    return linever, linetop, linebot, minus

def right_bracket(x, y, index):

    linever = plt.Line2D([x[index], x[index]], [y[0]-0.01, y[1]+0.01], color='black')
    linetop = plt.Line2D([x[index], x[index]-0.01], [y[1]+0.01, y[1]+0.01], color='black')
    linebot = plt.Line2D([x[index], x[index]-0.01], [y[0]-0.01, y[0]-0.01], color='black')
    
    eqtop = plt.Line2D([x[index] + 0.005, x[index] + 0.009], [0.505, 0.505], color='black')
    eqbot = plt.Line2D([x[index] + 0.005, x[index] + 0.009], [0.495, 0.495], color='black')

    return linever, linetop, linebot, eqtop, eqbot

def plus(x, index, fig):
    
    aspect = fig.get_figheight() / fig.get_figwidth()

    hor = plt.Line2D([x[index]-0.002, x[index]+0.002], [0.5, 0.5], color='black')
    ver = plt.Line2D([x[index], x[index]], [0.5-(0.002/aspect), 0.5+(0.002/aspect)], color='black')

    return hor, ver

def find_bbox(fig, axes):
    # Get the bounding boxes of the axes including text decorations
    
    r = fig.canvas.get_renderer()
    get_bbox = lambda ax: ax.get_tightbbox(r).transformed(fig.transFigure.inverted())
    xs_max = np.array([np.array(get_bbox(a), mtrans.Bbox)[1, 0] for a in axes])
    xs_min = np.array([np.array(get_bbox(a), mtrans.Bbox)[0, 0] for a in axes])
    xs_max = np.insert(xs_max, 0, 0)
    xs_min = np.append(xs_min, 1)
    xs = (xs_min + xs_max)/2

    ys = np.array([np.array(get_bbox(axes[0]), mtrans.Bbox)[0, 1], np.array(get_bbox(axes[0]), mtrans.Bbox)[1, 1]])

    return xs, ys



def plot_phase_closure(arr, loop_dates, dates, ml, save, plot): 
    """ 
    Func to plot the closure phase in radar coords and as a histogram. The mean amplitude is also plotted in radar coords. 
    """
    plt.style.use(['seaborn-poster'])    

    fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(12, 4), gridspec_kw=dict(width_ratios=[1/3, 1/3, 1/3]))
    amp = multilook(np.load("/home/jacob/phase_bias/mean_amp.npy"), ml[0], ml[1])
    ax[0].matshow(amp, vmax=3, cmap="binary_r")
    p = ax[1].matshow(arr, cmap='RdYlBu', vmin=-np.pi, vmax=np.pi)
    n = ax[2].hist(arr.flatten(), bins=np.linspace(-np.pi, np.pi, 50), alpha=0.8)[0]
    ax[2].text(-0.5, np.max(n)*0.7, f"$\mu$ = {np.mean(arr):.2f}", horizontalalignment='right')
    ax[2].axvline(x=0, color='black')
    
    ax[0].get_shared_x_axes().join(ax[0], ax[1])
    ax[0].get_shared_y_axes().join(ax[0], ax[1])

    for a in ax:
        a.grid()
        a.axis("off")

    ax[0].set_title("Mean amplitude")
    ax[1].set_title("Closure phase")
    ax[2].set_title("Closure phase histogram")

    plt.suptitle(loop_dates, y=0.08)
    cbar = plt.colorbar(p, ax=ax[-1], orientation='horizontal')
    cbar.ax.set_ylabel("Closure phase (radians)")

    save_fn = f"12_6_6/ML12_48/{loop_dates.split(' ')[-3]}_{loop_dates.split(' ')[-1]}.png"
    if save:
        plt.savefig(save_fn)
        np.save(save_fn[:-4] + ".npy", arr)
    else:
        pass
    if plot:
        plt.show()
    else:
        pass

    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(8, 4))
    p1 = ax[0].matshow(amp, vmin=0, vmax=3, cmap="binary_r")
    cbar1 = plt.colorbar(p1, ax=ax[0], shrink=0.8)
    cbar1.ax.set_ylabel("Amplitude", fontsize=18)
    cbar1.ax.set_yticks(range(4), [str(r) for r in range(4)])

    p2 = ax[1].matshow(arr, cmap="RdYlBu", vmin=-np.pi, vmax=np.pi)
    cbar2 = plt.colorbar(p2, ax=ax[1], shrink=0.8)
    cbar2.ax.set_ylabel("Closure phase (radians)", fontsize=18)
    cbar2.ax.set_yticks([-np.pi, 0, np.pi], ["$-\pi$", "0", "$\pi$"])
    
    ax[0].axis("off")
    ax[1].axis("off")
    fig.savefig("phase_closure_map.png") 


def parse_args():
    """
    Parses command line arguments and defines help output
    """
    parser = argparse.ArgumentParser(description='Compute and plot the closure phase.')
    parser.add_argument("-w",
                        dest="wdir", 
                        type=str,
                        help='Directory containing the single master interferogram phase hdf5 files',
                        default="/workspace/rapidsar_test_data/south_yorkshire/jacob2/")
    parser.add_argument("-i",
                        type=int,
                        dest='start_index',
                        default=0,
                        help='Start index of the loop')
    parser.add_argument("-d",
                        type=int,
                        dest='loop_delta',
                        default=2,
                        help='Delta for the jumpy (ie. 2 for a triplet.')
    parser.add_argument("-m",
                        type=str,
                        dest='multilook',
                        help="Multilook factor",
                        default='3,12')
    parser.add_argument("-s",
                        type=int,
                        dest="save",
                        help="Boolean to save (True) or not save (False).",
                        default=1)
    parser.add_argument("-p",
                        type=int,
                        dest="plot",
                        help="Boolean to show plot (True) or not show plot (False).",
                        default=1)
    args = parser.parse_args()
    return vars(args)

def getFiles(wdir):
    
    fns = [file for file in glob.glob(str(wdir), recursive=True)]

    fns.sort()
    
    return fns

def main():
    args_dict = parse_args()
    wdir = str(args_dict["wdir"])
    start = int(args_dict["start_index"])
    delta = int(args_dict["loop_delta"])
    ml = np.array(args_dict["multilook"].split(','), dtype=int)
    save_bool = bool(args_dict["save"])
    plot = bool(args_dict["plot"])
    # Make a list of all the images
    im_path = wdir + 'IFG/singlemaster/*/*.*'
    
    im_fns = [file for file in glob.glob(str(im_path), recursive=True)]

    im_fns.sort() # Sort the images by date

    # Use the image fns to get the date of each secondary SLC
    dates = [datetime.strptime(d.split('_')[-2], '%Y%m%d') for d in im_fns]

    # start, delta, ml1, ml2 = np.array(sys.argv[1:], dtype=int)
    end = int(start) + int(delta)
    ifgs, ifg_span = create_loop(start, end, dates, im_fns, ml)
    # Formatting a title for the figure
    loop_days = [str((dates[di+1] - dates[di]).days) for di in np.arange(start, end, 1)]
    loop_dates = f"{(dates[end]-dates[start]).days} - (" + ', '.join(loop_days) + f") | ML = [{ml[0]}, {ml[1]}] \n\
        {dates[start].strftime('%Y%m%d')} to {dates[end].strftime('%Y%m%d')}"
    arr = phase_closure(ifgs, ifg_span)
    print (loop_dates)
    plot_phase_closure(arr, loop_dates, dates, ml, save_bool, plot)
    
    plt.style.use(['seaborn-poster'])    
    plot_fig2(ifgs, ifg_span, arr)

if __name__ == "__main__":
    sys.exit(main())