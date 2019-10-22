import torch
torch.manual_seed(0)
from torch import nn
import torch.multiprocessing as mp
import torch.utils.data
from torch.autograd import Variable
import numpy as np

from model import FCN
from dataset import Dataset

import argparse
import time
from os.path import dirname, abspath, join
cur_dir = dirname(abspath(__file__))


parser = argparse.ArgumentParser()
parser.add_argument('--async', action="store", dest="async", type=bool, default=False,
                    help="Type True/False to turn on/off the async SGD. Default False")
parser.add_argument("--process", action="store", dest="process", type=int, default=4,
                    help="Number of processes to use if asynchronous SGD is turned on. Default 4")
parser.add_argument('--MH', action="store", dest="MH", type=bool, default=False,
                    help="Type True/False to use MinHash/feature hashing files as input. Default False")
parser.add_argument('--K', action="store", dest="K", type=int, default=1000,
                    help="K minhashes to use. The corresponding minhash file should be generated already. Default 1000")
parser.add_argument('--L', action="store", dest="L", type=int, default=3,
                    help="L layers of fully connected neural network to use. Default 3")
parser.add_argument('--dataset', action="store", dest="dataset", default="rcv1",
                    help="Dataset folder to use. Default rcv1")
parser.add_argument('--epoch', action="store", dest="epoch", type=int, default=10,
                    help="Number of epochs for training. Default 10")
parser.add_argument('--batch', action="store", dest="batch_size", type=int, default=100,
                    help="Batch size to use. Default 100")

results = parser.parse_args()


# ===========================================================
# Global variables & Hyper-parameters
# ===========================================================
DATASET = results.dataset
ASYNC = results.async
PROCESS = results.process
MH = results.MH
K = results.K
L = results.L
EPOCH = results.epoch
BATCH_SIZE = results.batch_size
GPU_IN_USE = True  # whether using GPU

with open(join(cur_dir, DATASET, "data", "dim.txt"), "r+") as infile:
    D = int(infile.readline())


def train(data_files, dim, model, time_file=None, record_files=None, p_id=None):
    # ===========================================================
    # Prepare train dataset & test dataset
    # ===========================================================
    print("***** prepare data ******")
    X_train, y_train, X_test, y_test = data_files
    if record_files is not None:
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
    training_time = 0.
    for epoch in range(EPOCH):
        # Training
        for iteration, (x, y) in enumerate(train_dataloader):
            start = time.clock()
            model.train()
            x = Variable(x).cuda() if GPU_IN_USE else Variable(x)
            y = Variable(y).cuda() if GPU_IN_USE else Variable(y)

            output = model(x)
            loss = loss_func(output, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            training_time += time.clock() - start
            _, predicted = torch.max(output.data, 1)
            train_accuracy = (predicted == y.data).sum().item() / y.data.shape[0]
            if iteration % 5:
                acc_list.append(train_accuracy)
                loss_list.append(loss.data)
            print('Epoch: ', epoch, '| Iteration: ', iteration, '| train loss: %.4f' % loss.data,
                  '| train accuracy: %.2f' % train_accuracy)

        valid_acc, valid_loss = validation(model, validation_dataloader, loss_func)
        valacc_list.append(valid_acc)
        valloss_list.append(valid_loss)
        print("Epoch:  {} -- validation accuracy: {} | validation loss: {}".format(epoch, valid_acc, valid_loss))
        print("*" * 50)

        # Saving records
        if record_files is not None:
            np.savetxt(acc_name, acc_list)
            np.savetxt(valacc_name, valacc_list)
            np.savetxt(loss_name, loss_list)
            np.savetxt(valloss_name, valloss_list)

    if time_file is not None:
        with open(time_file, 'a+') as outfile:
            prefix = "(ASYNC, id={}) ".format(p_id) if ASYNC else ""
            if MH:
                outfile.write("{}K={},   L={}, epoch={} | time={}\n".format(prefix, K, L, EPOCH, training_time))
            else:
                outfile.write("{}Baseline, L={}, epoch={} | time={}\n".format(prefix, L, EPOCH, training_time))


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
    print("dataset={}; async={}; num_process={} ; MH={}; K={}; L={}; epoch={}; batch_size={}".format(DATASET, ASYNC, PROCESS, MH, K, L, EPOCH, BATCH_SIZE))

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
    data_dirs = list(map(lambda f: join(cur_dir, DATASET, "data", f), data_files))
    print(data_files)
    time_file = join(cur_dir, DATASET, "record", "time_record.txt")

    if not ASYNC:
        record_files = ["acc{}_L{}.txt".format(fix, L), "val_acc{}_L{}.txt".format(fix, L),
                        "loss{}_L{}.txt".format(fix, L), "val_loss{}_L{}.txt".format(fix, L)]
        record_dirs = list(map(lambda f: join(cur_dir, DATASET, "record", f), record_files))
        train(data_dirs, D, model, time_file, record_dirs)
    else:
        mp.set_start_method('spawn')
        model.share_memory()
        processes = []
        all_record_dirs = []
        for p_id in range(PROCESS):
            record_files = ["pid{}_acc{}_L{}.txt".format(p_id, fix, L), "pid{}_val_acc{}_L{}.txt".format(p_id, fix, L),
                            "pid{}_loss{}_L{}.txt".format(p_id, fix, L), "pid{}_val_loss{}_L{}.txt".format(p_id, fix, L)]
            record_dirs = list(map(lambda f: join(cur_dir, DATASET, "record", f), record_files))
            all_record_dirs.append(record_dirs)
            p = mp.Process(target=train, args=(data_dirs, D, model),
                           kwargs={"time_file": time_file, "record_files": record_dirs, "p_id": p_id})
            p.start()
            processes.append(p)
        for p in processes:
            p.join()

        # File combination
        acc, valacc, loss, valloss = [], [], [], []
        for fnames in all_record_dirs:
            acc.append(np.loadtxt(fnames[0]))
            valacc.append(np.loadtxt(fnames[1]))
            loss.append(np.loadtxt(fnames[2]))
            valloss.append(np.loadtxt(fnames[3]))
        acc = np.mean(np.array(acc), axis=0).ravel()
        valacc = np.mean(np.array(valacc), axis=0).ravel()
        loss = np.mean(np.array(loss), axis=0).ravel()
        valloss = np.mean(np.array(valloss), axis=0).ravel()

        acc_name = join(cur_dir, DATASET, "record", "[ASYNC]acc{}_L{}.txt".format(fix, L))
        valacc_name = join(cur_dir, DATASET, "record", "[ASYNC]val_acc{}_L{}.txt".format(fix, L))
        loss_name = join(cur_dir, DATASET, "record", "[ASYNC]loss{}_L{}.txt".format(fix, L))
        valloss_name = join(cur_dir, DATASET, "record", "[ASYNC]val_loss{}_L{}.txt".format(fix, L))
        np.savetxt(acc_name, acc)
        np.savetxt(valacc_name, valacc)
        np.savetxt(loss_name, loss)
        np.savetxt(valloss_name, valloss)
