import torch
import torch.nn as nn
import torch.nn.functional as F
import time
from math import ceil

start_time = time.time()
random_seed = 177
GPU_available = False #torch.cuda.is_available()  #is GPU available?
device = "cuda" if GPU_available else "cpu"

# Processing parameters
num_samples = 25000        # how many times is each chromosome sampled?
num_chrom   = 100        # how many different chromosomes do we have access to?
num_classes = 3


# Training parameters
panel_dir = "../../simulate_admixture/panels/"
phase_dir = "../../simulate_admixture/phases/"
num_files =  390      #number of files to process in training
num_epochs = 300
batch_size = 4
train_prop = 0.9
eval_interval = 20   #how many epocchs to train before estimating loss 
num_estimate = 500    #number of examples to estimate accuracy with
lr_start = 3e-4
lr_end = lr_start/100
mixed_precision = False #train with mixed floating point precision?  helps with GPU memory issues
checkpointing = False

# Model parameters
input_size_processing = 100
input_size_step = 2
n_embd_processing = 475

input_size = input_size_processing
n_embd = n_embd_processing - 1

head_size = 60
num_heads = 4 #head_size must be divisible by num_heads
num_blocks = 3
dropout = 0.0

assert head_size % num_heads == 0
