import matplotlib.pyplot as plt
import numpy as np
from os.path import dirname, abspath, join
cur_dir = dirname(abspath(__file__))
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--K', action="append", dest="K", type=int, default=[])
parser.add_argument('--L', action="store", dest="L", type=int, default=3)

results = parser.parse_args()


def plot_experiment(K_arr, L, metric="acc", baseline=True):
    title = "{} (Layer={})".format(metric, L)
    record_dir = join(cur_dir, "rcv1", "record")

    if baseline:
        baseline_acc = np.loadtxt(join(record_dir, "{}_L{}.txt".format(metric, L)))
        baseline_val = np.loadtxt(join(record_dir, "val_{}_L{}.txt".format(metric, L)))
        plt.plot(np.arange(1, baseline_acc.shape[0] + 1), baseline_acc, label="Train(baseline)", color="r")
        plt.plot(np.arange(1, baseline_val.shape[0] + 1), baseline_val, label="Test(baseline)", color="g")
        fig_name = join(record_dir, "pic_{}_L{}.png".format(metric, L))

    for k in K_arr:
        acc = np.loadtxt(join(record_dir, "{}_K{}_L{}.txt".format(metric, k, L)))
        val_acc = np.loadtxt(join(record_dir, "val_{}_K{}_L{}.txt".format(metric, k, L)))
        plt.plot(np.arange(1, acc.shape[0] + 1), acc, label="Train(K={})".format(k))
        plt.plot(np.arange(1, val_acc.shape[0] + 1), val_acc, label="Test(K={})".format(k))
        fig_name = join(record_dir, "pic_{}_K{}_L{}.txt".format(metric, k, L))

    plt.title(title)
    plt.xlabel("Epoch")
    plt.legend()
    plt.savefig(fig_name, bbox_inches='tight')
    plt.close()


if __name__ == '__main__':
    K_arr = results.K
    L = results.L
    plot_experiment(K_arr, L, metric="acc")
    plot_experiment(K_arr, L, metric="loss")
