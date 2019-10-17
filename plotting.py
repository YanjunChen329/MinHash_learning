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
    fig = plt.figure(figsize=(15, 5))

    if baseline:
        baseline_acc = np.loadtxt(join(record_dir, "{}_L{}.txt".format(metric, L)))
        baseline_val = np.loadtxt(join(record_dir, "val_{}_L{}.txt".format(metric, L)))

        step = int(baseline_acc.shape[0] / baseline_val.shape[0])
        train_axis = np.arange(1, baseline_acc.shape[0] + 1)
        val_axis = np.arange(step, baseline_acc.shape[0] + step, step)

        plt.plot(train_axis, baseline_acc, label="Train(baseline)", color="r", zorder=1)
        plt.plot(val_axis, baseline_val, label="Test(baseline)", color="g", zorder=2)
        fig_name = join(record_dir, "pic_{}_L{}.png".format(metric, L))

    for k in K_arr:
        acc = np.loadtxt(join(record_dir, "{}_K{}_L{}.txt".format(metric, k, L)))
        val_acc = np.loadtxt(join(record_dir, "val_{}_K{}_L{}.txt".format(metric, k, L)))

        step = int(acc.shape[0] / val_acc.shape[0])
        train_axis = np.arange(1, baseline_acc.shape[0] + 1)
        val_axis = np.arange(step, baseline_acc.shape[0] + step, step)

        plt.plot(train_axis, acc, label="Train(K={})".format(k), zorder=1)
        plt.plot(val_axis, val_acc, label="Test(K={})".format(k), zorder=2)
        fig_name = join(record_dir, "pic_{}_K{}_L{}.png".format(metric, k, L))

    plt.title(title)
    plt.xlabel("Iteration")
    plt.legend()
    fig.savefig(fig_name, bbox_inches='tight')
    plt.close()


if __name__ == '__main__':
    K_arr = results.K
    L = results.L
    plot_experiment(K_arr, L, metric="acc")
    plot_experiment(K_arr, L, metric="loss")
