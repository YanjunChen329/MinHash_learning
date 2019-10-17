import torch
torch.manual_seed(0)
from torch import nn
import torch.utils.data
from torch.autograd import Variable
import numpy as np
import argparse

from model import FCN
from dataset import Dataset

from os.path import dirname, abspath, join
cur_dir = dirname(abspath(__file__))


parser = argparse.ArgumentParser()
parser.add_argument('--MH', action="store", dest="MH", type=bool, default=False)
parser.add_argument('--K', action="store", dest="K", type=int, default=1000)
parser.add_argument('--L', action="store", dest="L", type=int, default=3)
parser.add_argument('--dataset', action="store", dest="dataset", default="rcv1")
parser.add_argument('--epoch', action="store", dest="epoch", type=int, default=20)
parser.add_argument('--batch', action="store", dest="batch_size", type=int, default=100)

results = parser.parse_args()


# ===========================================================
# Global variables & Hyper-parameters
# ===========================================================
dataset = results.dataset
MH = results.MH
K = results.K
L = results.L
EPOCH = results.epoch
BATCH_SIZE = results.batch_size
GPU_IN_USE = True  # whether using GPU

with open(join(cur_dir, dataset, "data", "dim.txt"), "r+") as infile:
    D = int(infile.readline())


def train(data_files, dim, model, record_files):
    # ===========================================================
    # Prepare train dataset & test dataset
    # ===========================================================
    print("***** prepare data ******")
    X_train, y_train, X_test, y_test = data_files
    acc_name, valacc_name, loss_name, valloss_name = record_files

    training_set = Dataset(X_train, y_train, dimension=dim)
    train_dataloader = torch.utils.data.DataLoader(dataset=training_set, batch_size=BATCH_SIZE, shuffle=False)
    validation_set = Dataset(X_test, y_test, dimension=dim)
    validation_dataloader = torch.utils.data.DataLoader(dataset=validation_set, batch_size=BATCH_SIZE, shuffle=False)

    print("***** prepare optimizer ******")
    optimizer = torch.optim.Adam(model.parameters())
    loss_func = nn.CrossEntropyLoss().cuda() if GPU_IN_USE else nn.CrossEntropyLoss()

    print("***** Train ******")
    acc_list, valacc_list = [], []
    loss_list, valloss_list = [], []

    for epoch in range(EPOCH):
        # Training
        for iteration, (x, y) in enumerate(train_dataloader):
            model.train()
            x = Variable(x).cuda() if GPU_IN_USE else Variable(x)
            y = Variable(y).cuda() if GPU_IN_USE else Variable(y)

            output = model(x)
            loss = loss_func(output, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            _, predicted = torch.max(output.data, 1)
            train_accuracy = (predicted == y.data).sum().item() / y.data.shape[0]
            acc_list.append(train_accuracy)
            loss_list.append(loss.data)
            print('Epoch: ', epoch, '| Iteration: ', iteration, '| train loss: %.4f' % loss.data,
                  '| train accuracy: %.2f' % train_accuracy)

        valid_acc, valid_loss = validation(model, validation_dataloader, loss_func)
        valacc_list.append(valid_acc)
        valloss_list.append(valid_loss)
        print("Epoch:  {} -- validation accuracy: {} | validation loss: {}".format(epoch, valid_acc, valid_loss))
        print("*" * 50)

        # Saving
        np.savetxt(acc_name, acc_list)
        np.savetxt(valacc_name, valacc_list)
        np.savetxt(loss_name, loss_list)
        np.savetxt(valloss_name, valloss_list)


def validation(model, validation_dataloader, loss_func):
    count = 0.
    total = 0.
    valid_correct = 0.
    total_loss = 0.
    model.eval()

    with torch.no_grad():
        for _, (x_t, y_t) in enumerate(validation_dataloader):
            x_t = Variable(x_t).cuda() if GPU_IN_USE else Variable(x_t)
            y_t = Variable(y_t).cuda() if GPU_IN_USE else Variable(y_t)

            output = model(x_t)
            _, predicted = torch.max(output.data, 1)
            loss = loss_func(output, y_t)
            total_loss += loss
            valid_correct += (predicted == y_t.data).sum().item()
            total += y_t.data.shape[0]
            count += 1

        valid_accuracy = valid_correct / total
        valid_loss = total_loss / count

    return valid_accuracy, valid_loss


if __name__ == '__main__':
    print("dataset={}; MH={}; K={}; L={}; epoch={}; batch_size={}".format(dataset, MH, K, L, EPOCH, BATCH_SIZE))

    #########################################
    print("***** prepare model ******")
    model = FCN(dimension=D, num_layers=L).double()
    if GPU_IN_USE:
        model.cuda()
    # print(torch.cuda.device_count())
    print(model)
    #########################################

    fix = "_K{}".format(K) if MH else ""
    data_files = ["train{}_X.txt".format(fix), "train_y.txt", "test{}_X.txt".format(fix), "test_y.txt"]
    data_dirs = list(map(lambda f: join(cur_dir, dataset, "data", f), data_files))
    print(data_files)
    record_files = ["acc{}_L{}.txt".format(fix, L), "val_acc{}_L{}.txt".format(fix, L),
                    "loss{}_L{}.txt".format(fix, L), "val_loss{}_L{}.txt".format(fix, L)]
    record_dirs = list(map(lambda f: join(cur_dir, dataset, "record", f), record_files))

    train(data_dirs, D, model, record_dirs)
