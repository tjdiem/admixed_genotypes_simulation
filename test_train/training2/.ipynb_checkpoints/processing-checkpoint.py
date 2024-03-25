import models
import os
import subprocess
import time
import random
from math import sqrt, e, pi
import numpy as np
import matplotlib.pyplot as plt

random_seed = 177
num_samples = models.len_chrom
num_chrom = 100
start_time = time.time()
sample_width = models.sample_width

random.seed(random_seed)

sigma = 2.5

assert sample_width >= 1 and sample_width % 2 == 1

def GetMemory():
    if os.name == 'posix':
        mem_info = subprocess.check_output(['free','-b']).decode().split()
        total_memory = int(mem_info[7]) - int(mem_info[8]) 
        total_memory *= 10**-9
        
    elif os.name == 'nt':
        mem_info = subprocess.check_output(['wmic','OS','get','FreePhysicalMemory']).decode().split()
        total_memory = int(mem_info[1]) * 1024 * 10**-9
        
    print(f"Available memory: {total_memory:0.2f} GB")
    
def GetTime():
    seconds = int(time.time() - start_time)
    hours = seconds // 3600
    seconds = seconds % 3600
    minutes = seconds // 60
    seconds = seconds % 60
    print(f"Total time elapsed: {hours}h {minutes}m {seconds}s")

Normal = lambda x, mean: e**(-0.5*((x-mean)/sigma)**2) / (sigma * sqrt(2*pi))

def split_to_sample(split_file, sampled_sites):
    with open(split_file, "r") as f:
        lines = f.readlines()

    lines = [[float(l) for l in line] for line in lines]

    min_site = sampled_sites[0]

    X = [[0 for _ in range(len(lines)//2)] for _ in range(len(sampled_sites))]
    for i in range(0,len(lines),2):
        split_points = lines[i]
        categories = lines[i+1]

        # find correct starting index
        ind = min_site * len(split_points)
        if split_points[ind] < min_site:  
            while split_points[ind] < min_site:
                ind += 1
            ind -= 1
        else:
            while split_points[ind] >= min_site:
                ind -= 1

        # sample sites
        for j,site in enumerate(sampled_sites):
            while site > split_points[ind+1]:
                ind += 1

            X[j][i // 2] = categories[ind]




def convert_files(sampling_file, command_file):

    if command_file.endswith("000"):
        print(command_file[-5:])

    with open(sampling_file, "r") as f:
        lines = f.readlines()

    if len(lines) < 100:
        return None


    X = [[int(l) for l in line[:-1]] for line in lines]

    #random.shuffle(out) #This will shuffle the rows
    with open(command_file, "r") as f:
        s = f.readlines()[0].split()

    if -0.01 < float(s[11]) < 0.01:
        return None
    
    if float(s[8]) < 0.05:
        return None

    # if 0.4 <= float(s[2]) <= 0.6:
    #     return None

    points = [float(s[6]), float(s[7]), float(s[10])]
    ind1, ind2, regular_ind = [round(num_samples*point - 0.5) for point in points]

    site1 = [sample[ind1] for sample in X]
    site2 = [sample[ind2] for sample in X]
    regular_site = [sample[regular_ind] for sample in X]

    for offset in range(1, sample_width // 2 + 1):
        site_prev = [sample[ind1 - offset] for sample in X] if ind1 - offset >= 0 else [1 for _ in range(len(X))]
        site_next = [sample[ind1 + offset] for sample in X] if ind1 + offset < num_samples else [1 for _ in range(len(X))]
        site1 = site_prev + site1 + site_next

        site_prev = [sample[ind2 - offset] for sample in X] if ind2 - offset >= 0 else [1 for _ in range(len(X))]
        site_next = [sample[ind2 + offset] for sample in X] if ind2 + offset < num_samples else [1 for _ in range(len(X))]
        site2 = site_prev + site2 + site_next

        site_prev = [sample[regular_ind - offset] for sample in X] if regular_ind - offset >= 0 else [1 for _ in range(len(X))]
        site_next = [sample[regular_ind + offset] for sample in X] if regular_ind + offset < num_samples else [1 for _ in range(len(X))]
        regular_site = site_prev + regular_site + site_next

    out_true = site1 + site2 if random.random() < 0.5 else site2 + site1

    ep_site = site1 if random.random() < 0.5 else site2
    # ep_site = site2

    out_false = ep_site + regular_site if random.random() < 0.5 else regular_site + ep_site

    return [out_true, out_false]

def convert_command_file1(file):
    
    with open(file,"r") as f:
        lines, = f.readlines()
    

    flist = [float(x) for x in lines.split()[1:]]

    return [flist,flist]