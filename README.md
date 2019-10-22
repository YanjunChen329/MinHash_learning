# MinHash Learning

The research project explores using MinHash in replace of feature hashing to train neural networks on high-dimensional sparse data. Because the power law of the data distribution, we hypothesized that using MinHash allows the model to converge faster and better, especially when the asynchronous SGD Hogwild! is used.

## Prerequisites

This project is built with Python 3. It is using Pytorch 1.3 with CUDA 10.1. Besides, it also uses some common data science libraries such as sklearn, numpy and matplotlib

## Running Experiments

This project is built for training with arbitrary dataset. So far, the repo only supports the RCV1 binary dataset. 
For RCV1, the feature-hashed data are already pre-processed (using the script rcv1/data_preprocessing.py) and stored in the repo, under the folder /rcv1/data. Thus, users don't need to download anything from the web other than cloning the repo.
To run an experiment, the following 3 steps should be followed:

### 1. Generating MinHash files

Users should generate files of minhashed data prior to doing experiments. To generate minhash files, run generate_minhash.py under the folder ./rcv1 followed by a K value. The K value is the number of minhashes you want to generate. An example command is as followed:
```
$ python3 ./rcv1/generate_minhash.py 2000
```
It seems that using K=2000 gives good result. Users may want to stick to it.


### 2. Training the neural network

After getting all the data, users should use training.py under the root directory to train the model and save the training record. There are several arguments that can be passed into training.py. Users can run the following command to get a help message descirbing the usage:
```
$ python3 training.py -h

usage: training.py [-h] [--async ASYNC] [--process PROCESS] [--MH MH] [--K K]
                   [--L L] [--dataset DATASET] [--epoch EPOCH]
                   [--batch BATCH_SIZE]

optional arguments:
  -h, --help          show this help message and exit
  --async ASYNC       Type True/False to turn on/off the async SGD. Default
                      False
  --process PROCESS   Number of processes to use if asynchronous SGD is turned
                      on. Default 4
  --MH MH             Type True/False to use MinHash/feature hashing files as
                      input. Default False
  --K K               K minhashes to use. The corresponding minhash file
                      should be generated already. Default 1000
  --L L               L layers of fully connected neural network to use.
                      Default 3
  --dataset DATASET   Dataset folder to use. Default rcv1
  --epoch EPOCH       Number of epochs for training. Default 10
  --batch BATCH_SIZE  Batch size to use. Default 100
```

### 3. Plotting the result

After running the experiments, users can use plotting.py under the root directory to plot and save the training results. Like training.py, users can run the following command to get a help message descirbing the usage:
```
$ python3 plotting.py -h

usage: plotting.py [-h] [--async ASYNC] [--K K] [--L L]

optional arguments:
  -h, --help     show this help message and exit
  --async ASYNC  Type True/False to plot async SGD/normal training. Default
                 False
  --K K          K-minhash file to plot. The corresponding experiment should
                 be run already
  --L L          L-layer FCNN to plot. Default 3
```
Before plotting on any given ASYNC, K and L, user should run experiments with the same ASYNC, K and L using training.py. Note that plotting.py would in default plot the minhash result with feature hashing result, so be sure to run both experiments before plotting with a K. If users left the K argument unwritten, the script will only plot results from feature hashing. The plot will be saved under ./rcv1/record/

