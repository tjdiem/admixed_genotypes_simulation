import torch
import torch.nn as nn
import torch.nn.functional as F
import time


start_time = time.time()
random_seed = 177
GPU_available = torch.cuda.is_available()  #is GPU available?
device = "cuda" if GPU_available else "cpu"


# Processing parameters
num_samples = 500        # how many times is each chromosome sampled?
num_chrom   = 100        # how many different chromosomes do we have access to?


# Training parameters
data_dir = "../Data1"                      #folder the data is in
num_files = 25000       #number of files to process in training
num_epochs = 30
batch_size = 32
train_prop = 0.9
num_estimate = 5000      #number of examples to estimate accuracy with
lr_start = 6e-5
lr_end = lr_start/100


# Model parameters
# input_size = num_samples + 1
# n_embd = num_chrom + 1

input_size = num_chrom
n_embd = num_samples + 1

head_size = 192
num_heads = 6  #head_size must be divisible by num_heads
num_blocks = 4
dropout = 0.15

assert head_size % num_heads == 0



