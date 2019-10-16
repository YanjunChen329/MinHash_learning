import sys
from os.path import dirname, abspath, join

cur_dir = dirname(abspath(__file__))
print(cur_dir)

infilename = join(cur_dir, "data", "rcv1_test.binary")
trainX_name = join(cur_dir, "data", "train_X.txt")
testX_name = join(cur_dir, "data", "test_X.txt")
trainy_name = join(cur_dir, "data", "train_y.txt")
testy_name = join(cur_dir, "data", "test_y.txt")

counter, stop = 0, 50000
train_test_ratio = 0.2
mod = int(1 / train_test_ratio)
dim = 0
label_size = [0, 0]

with open(infilename) as inputfile, open(trainX_name, "w+") as train_X, open(trainy_name, "w+") as train_y, open(
        testX_name, "w+") as test_X, open(testy_name, "w+") as test_y:
    for line in inputfile:
        sys.stdout.write("\r{:.3f}%".format(100. * counter / stop))
        sys.stdout.flush()

        arr = line.split(" ")
        label, data = arr[0], arr[1:]
        if label == "-1":
            label_size[0] += 1
            label = "0"
        else:
            label_size[1] += 1
            label = "1"

        vector = list(map(lambda x: x.split(":")[0], data))
        dim = int(vector[-1]) if int(vector[-1]) > dim else dim
        if counter % mod == 0:
            test_X.write(",".join(vector) + "\n")
            test_y.write(label + "\n")
        else:
            train_X.write(",".join(vector) + "\n")
            train_y.write(label + "\n")
        counter += 1
        if counter >= stop:
            break
print()
print(dim)
print(label_size)

with open("dim.txt", "w+") as infile:
    infile.write(dim)
