import numpy as np
import matplotlib.pyplot as plt
from loopClosure import getFiles, create_loop
import sys
from datetime import datetime
import argparse


def parse_args():
    """
    Parses command line arguments and defines help output
    """
    parser = argparse.ArgumentParser(description='Compute and plot the closure phase.')
    parser.add_argument("-w",
                        dest="wdir", 
                        type=str,
                        help='Directory containing the triplet loop closure.',
                        default='/home/jacob/phase_bias/12_6_6/ML12_48/data/*.npy')
    parser.add_argument("-s",
                        type=bool,
                        dest="save",
                        help="Boolean to save (True) or not save (False).",
                        default=0)
    parser.add_argument("-p",
                        type=bool,
                        dest="plot",
                        help="Boolean to show plot (True) or not show plot (False).",
                        default=1)
    args = parser.parse_args()
    return vars(args)


def main():
    args_dict = parse_args()
    wdir = str(args_dict["wdir"])
    fns = getFiles(wdir)
    save = args_dict["save"]
    plot = args_dict["plot"]
    # dates = [datetime.strptime(d.split('_')[-2], '%Y%m%d') for d in fns]
    shape = np.load(fns[0]).shape
    
    cumulative = np.empty((len(fns[::1]), *shape))
    for i, fn in enumerate(fns[::1]):
        a = np.load(fn)
        cumulative[i] = a
    print (cumulative.shape)
    if plot:
        p = plt.matshow(np.sum(cumulative, axis=0), cmap="RdYlBu")
        # p = plt.matshow(np.angle(np.mean(np.exp(1j*cumulative), axis=0)))
        plt.colorbar(p)
    if save:
        plt.savefig("test.png")
    
    plt.figure()
    plt.hist(np.sum(cumulative, axis=0).flatten())
    # plt.hist(np.angle(np.mean(np.exp(1j*cumulative), axis=0)).flatten(), bins=np.linspace(-np.pi, np.pi, 40))
    
    plt.show()
    
    

if __name__ == "__main__":
    sys.exit(main())