import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import sys
from math import ceil

"""
same as training8 but for diploid data
"""

start_time = time.time()
random_seed = 177
GPU_available = torch.cuda.is_available()
device = "cuda" if GPU_available else "cpu" 

# Processing parameters
num_samples = 25000        # how many times is each chromosome sampled?
num_chrom   = 100        # how many different chromosomes do we have access to?
num_classes = 3

# Training parameters
panel_template_dir = "../../data_simulated/panel_templates11/"
panel_dir = "../../data_simulated/panels11/"
split_dir = "../../data_simulated/splits11/"
parameters_dir = "../../data_simulated/parameters11/"

if sys.argv[0] == "train.py":
    num_files =  270   #number of files to process in training
    num_epochs = 3000
    batch_size = 16
    train_prop = 0.9
    eval_interval = 100   #how many epochs to train before evaluating model
    num_estimate = 1000    #number of examples to estimate accuracy with
    lr_start = 3e-4
    lr_end = lr_start/100
    mixed_precision = False #train with mixed floating point precision?  helps with GPU memory issues
    checkpointing = False

elif sys.argv[0] == "train_probs.py":
    num_files =  270   #number of files to process in training
    num_epochs = 3000
    batch_size = 1
    train_prop = 0.9
    eval_interval = 50   #how many epochs to train before evaluating model
    num_estimate = 200    #number of examples to estimate accuracy with
    lr_start = 3e-13
    lr_end = lr_start/100
    mixed_precision = False #train with mixed floating point precision?  helps with GPU memory issues
    checkpointing = False

    max_num_probs = 4
    start_model = "model_0.pth"

# Processing parameters
num_bp = 50_000_000          # total number of base pairs
input_size_processing = 501  # total number of SNPs in processing
input_size_step = 2          # step between SNPs
n_ind_adm = 400              # number of individuals in panel file
n_ind_pan = 50               # number of individuals in each reference panel


# Model parameter
n_ind_pan_model = n_ind_pan // num_classes
n_ind_max = n_ind_pan_model * num_classes
input_size = input_size_processing  # number of SNPs
# n_embd = n_ind_adm - 1      # number of individuals

sigma = 0.01

num_heads = 4 #head_size must be divisible by num_heads
head_size = 2 * num_heads * input_size
num_blocks = 3
dropout = 0.0

assert head_size % num_heads == 0
assert input_size % 2 == 1

