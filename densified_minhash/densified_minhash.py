from sklearn.utils import murmurhash3_32 as mmh
import numpy as np


class Densified_MinHash():
    def __init__(self, K, D, bbit=8, seed=0):
        self.K = K
        self.D = D
        self.bbit = bbit
        self.seed = seed

    def hash_func(self, seed):
        return lambda x: mmh(key=x, seed=seed, positive=True) % self.D

    def hash_bin_to_bin(self, bin_id, attempt, seed):
        key = str(attempt) + "." + str(bin_id)
        return mmh(key=key, seed=seed, positive=True) % self.K

    def convert_to_bit_array(self, k_hashes):
        one_hot = np.zeros((self.K, self.D))
        for i in range(self.K):
            one_hot[i, int(k_hashes[i])] = 1
        return one_hot.ravel()

    def get_hashed(self, word_set):
        k_hashes = [-1 for _ in range(self.K)]
        hash_func = self.hash_func(self.seed)
        for w in word_set:
            hash_val = hash_func(w)
            idx = int(1. * hash_val * self.K / self.D)
            if hash_val > k_hashes[idx]:
                k_hashes[idx] = hash_val

        # optimal densified hashing for empty bins
        for idx in range(self.K):
            if k_hashes[idx] == -1:
                attempt = 1
                new_bin = self.hash_bin_to_bin(idx, attempt, self.seed)
                while k_hashes[new_bin] == -1:
                    attempt += 1
                    new_bin = self.hash_bin_to_bin(idx, attempt, self.seed)
                k_hashes[idx] = k_hashes[new_bin]
        return k_hashes

if __name__ == '__main__':
    K = 10
    D = 10000
    s1 = ["a", "b", "c"]
    s2 = ["a", "b", "c", "d"]
    equal_hash = np.zeros(K)
    count = 0.
    for seed in range(10000):
        count += 1.
        DMH = Densified_MinHash(K, D, seed=seed)
        equal_hash += (np.array(DMH.get_hashed(s1)) == np.array(DMH.get_hashed(s2)))
        print(equal_hash / count)
