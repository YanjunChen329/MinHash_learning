# Generating minhash signatures
import sys
from os.path import dirname, abspath, join
cur_dir = dirname(abspath(__file__))
sys.path.append(dirname(cur_dir))

from densified_minhash.densified_minhash import Densified_MinHash


# get K
K = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
# get D
with open(join(cur_dir, "data", "dim.txt"), "r+") as infile:
    D = int(infile.readline())

DMH = Densified_MinHash(K=K, D=D)

trainX_name = join(cur_dir, "data", "train_X.txt")
testX_name = join(cur_dir, "data", "test_X.txt")
out_trainX = join(cur_dir, "data", "train_mh{}_X.txt".format(K))
out_testX = join(cur_dir, "data", "test_mh{}_X.txt".format(K))

print("K={}; D={}".format(K, D))

def generate_minhash(name, X_file, out_file):
    counter = 1
    with open(X_file, "r+") as infile, open(out_file, "w+") as outfile:
        print(name)
        for line in infile:
            sys.stdout.write("\rGenerating {}".format(counter))
            sys.stdout.flush()
            data = line.split(",")
            signatures = DMH.get_hashed(data)
            newline = ",".join(list(map(lambda x: str(x), signatures)))
            outfile.write(newline + "\n")
            counter += 1
        print()

generate_minhash("Training Set", trainX_name, out_trainX)
generate_minhash("Testing Set", testX_name, out_testX)
