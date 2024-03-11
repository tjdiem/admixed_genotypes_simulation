from processing import *
import numpy as np
import matplotlib.pyplot as plt
from ast import literal_eval
import random

# with open("smaller_epistatic.txt","r") as f:
#     idx = literal_eval(f.read())

idx = list(range(100_000))
random.shuffle(idx)

for i in idx[103:128]:

    sampling_file = "../sampled_genotypes/sample_" + str(i)
    with open(sampling_file, "r") as f:
        lines = f.readlines()

    X = [[float(l) for l in line[:-1]] for line in lines]

    X = np.array(X) - 1
    with open("../commands/command_" + str(i), "r") as f:
        s = f.readlines()[0].split()

    points = [float(s[6]), float(s[7])]
    inds = [round(num_samples*point - 0.5) for point in points]
    regular_site = round(num_samples*float(s[10]) - 0.5)


    X = X.sum(axis=0)

    plt.scatter(range(len(X)), X)
    plt.scatter(inds,X[inds])
    plt.scatter(regular_site,X[regular_site])
    plt.show()


