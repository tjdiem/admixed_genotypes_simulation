import torch
import torch.nn as nn
import torch.nn.functional as F
import time

start_time = time.time()
random_seed = 177
GPU_available = torch.cuda.is_available()  #is GPU available?
device = "cuda" if GPU_available else "cpu"

# Processing parameters
num_samples = 45642        # how many times is each chromosome sampled?
num_chrom   = 100        # how many different chromosomes do we have access to?
num_classes = 3


# Training parameters
panel_dir = "../../simulate_admixture/panels/"
phase_dir = "../../simulate_admixture/phases/"
num_files = 500       #number of files to process in training
num_epochs = 300
batch_size = 32
train_prop = 0.9
num_estimate = 5000      #number of examples to estimate accuracy with (defaults to test size if too large)
lr_start = 6e-5
lr_end = lr_start/100
mixed_precision = False #train with mixed floating point precision?  helps with GPU memory issues
checkpointing = False

# Model parameters
# input_size = num_samples + 1
# n_embd = num_chrom + 1

input_size_processing = 500
input_size_step = 15
n_embd_processing = 35

input_size = input_size_processing
n_embd = 2 * n_embd_processing + 2

head_size = 192
num_heads = 6  #head_size must be divisible by num_heads
num_blocks = 3
dropout = 0.0

assert head_size % num_heads == 0



