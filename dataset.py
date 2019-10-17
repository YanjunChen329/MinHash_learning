import torch
from torch.utils import data
import numpy as np


class Dataset(data.Dataset):
    def __init__(self, X_file, y_file, dimension):
        super(Dataset, self).__init__()
        with open(X_file, 'r+') as xfile:
            self.X = xfile.readlines()
        with open(y_file, 'r+') as yfile:
            self.y = yfile.readlines()
        self.length = len(self.X)
        self.dim = dimension

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        data_point = np.zeros(self.dim)
        nonzeros = self.X[index].replace("\n", "").split(",")
        label = int(self.y[index].replace("\n", ""))
        for idx in nonzeros:
            data_point[int(idx)] = 1.
        return data_point, label


if __name__ == '__main__':
    dataset = Dataset(X_file="./rcv1/data/train_X.txt", y_file="./rcv1/data/train_y.txt", dimension=50000)
    print(dataset[1])
