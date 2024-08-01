import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import time
import sys
from math import ceil

"""
same as training9 but with probabilities
"""

start_time = time.time()
random_seed = 177
GPU_available = torch.cuda.is_available()
device = "cuda" if GPU_available else "cpu"

# Processing parameters
num_samples = 25000        # how many times is each chromosome sampled?
num_chrom   = 100        # how many different chromosomes do we have access to?
num_classes = 3

model_name = "KNet4"

human_data = False
if human_data:
    panel_template_dir = "../../1000genomes_simulate/panel_templates/"
    panel_dir = "../../1000genomes_simulate/panels/"
    split_dir = "../../1000genomes_simulate/splits/"
    parameters_dir = "../../1000genomes_simulate/parameters/"
    n_ind_adm = 49
else:
    # Training parameters
    panel_template_dir = "../../data_simulated/panel_templates13/"
    panel_dir = "../../data_simulated/panels13/"
    split_dir = "../../data_simulated/splits13/"
    parameters_dir = "../../data_simulated/parameters13/"

if sys.argv[0] in ["train.py", "deep_clustering.py", "train_positions.py", "train_full_seq.py", "train_var_IS.py", "train_NG.py"] or sys.argv[0].startswith("predict_full_seq"): #change this later
    num_files = 600
    train_prop = 0.9
    num_epochs = 2500
    batch_size = 16
    eval_interval = 100    #how many epochs to train before evaluating model
    num_estimate = 500    #number of examples to estimate accuracy with
    lr_start = 3e-4
    lr_end = lr_start/200
    mixed_precision = False #train with mixed floating point precision?  helps with GPU memory issues
    checkpointing = False

elif sys.argv[0] == "train_probs.py":
    num_files = 270
    train_prop = 0.9
    num_epochs = 3000
    batch_size = 1
    eval_interval = 50   #how many epochs to train before evaluating model
    num_estimate = 200    #number of examples to estimate accuracy with
    lr_start = 3e-13
    lr_end = lr_start/100
    mixed_precision = False #train with mixed floating point precision?  helps with GPU memory issues
    checkpointing = False

    max_num_probs = 4
    start_model = "model_0.pth"

elif sys.argv[0] == "evaluate_model.py":
    num_files = 270
    train_prop = 0.9
    batch_size = 1

    
# Processing parameters
if sys.argv[0] == "deep_clustering.py":
    num_bp = 50_000_000                           # total number of base pairs
    input_size_processing = "all"                   # total number of SNPs in processing
    input_size_step = 1                           # step between SNPs
    n_ind_adm_start = 0
    n_ind_adm_end = 49 if human_data else 400     # number of individuals in panel file
    n_ind_pan = 50                                # number of individuals in each reference panel
elif sys.argv[0] in ["train_full_seq.py", "train_NG.py"] or sys.argv[0].startswith("predict_full_seq"):
    num_bp = 50_000_000                           # total number of base pairs
    input_size_processing = "all"                 # total number of SNPs in processing
    input_size_step = 1                           # step between SNPs
    n_ind_adm_start = 0
    n_ind_adm_end = 49 if human_data else 400         # number of individuals in panel file
    n_ind_pan = 50                                # number of individuals in each reference panel
else:
    num_bp = 50_000_000                           # total number of base pairs
    input_size_processing = 501                   # total number of SNPs in processing
    input_size_step = 2                           # step between SNPs
    n_ind_adm_start = 0
    n_ind_adm_end = 49 if human_data else 400         # number of individuals in panel file
    n_ind_pan = 50                                # number of individuals in each reference panel

if sys.argv[0].startswith("predict_full_seq"):
    n_ind_adm_start = 0
    n_ind_adm_end = 1

# Model parameter
n_ind_pan_model = n_ind_pan // num_classes
n_ind_max = n_ind_pan_model * num_classes
input_size = 501
input_size_positional = 1501
n_embd = 36
n_embd_model = n_embd + 2

# n_embd = n_ind_adm - 1      # number of individuals

sigma = 0.01

num_heads = 4 #head_size must be divisible by num_heads
head_size = 10 * num_heads
# head_size = 2 * num_heads * input_size
num_blocks = 3
dropout = 0.0

for argv in sys.argv:
    if "=" in argv:
        exec(argv)

assert head_size % num_heads == 0
assert input_size % 2 == 1

