from globals import *
from processing import *
import torch
import time

num_files = 367
chunk_size = 1000

min_freq_difference = 0.2
morgan_cutoff = 0.00002
base_pairs = 50000000
morgans = 1

with open("saved_inputs/file_to_length.txt", "r") as f:
    file_to_length = eval(f.read())

def load_filtered_file(file_num, filter_freq=True, filter_LD=True, idx_predict=None):

    X = torch.cat([torch.load(f"saved_inputs/X_file{file_num}_chunk{chunk}.pt") for chunk in range(0, file_to_length[file_num] // chunk_size * chunk_size + chunk_size, chunk_size)], dim=-1).long()
    y = torch.cat([torch.load(f"saved_inputs/y_file{file_num}_chunk{chunk}.pt") for chunk in range(0, file_to_length[file_num] // chunk_size * chunk_size + chunk_size, chunk_size)], dim=-1).long()
    refs = torch.cat([torch.load(f"saved_inputs/refs_file{file_num}_chunk{chunk}.pt") for chunk in range(0, file_to_length[file_num] // chunk_size * chunk_size + chunk_size, chunk_size)], dim=-1).float()
    positions = torch.cat([torch.load(f"saved_inputs/positions_file{file_num}_chunk{chunk}.pt") for chunk in range(0, file_to_length[file_num] // chunk_size * chunk_size + chunk_size, chunk_size)], dim=-1)
    params = torch.tensor(convert_parameter(parameters_dir + "parameter_" + str(file_num)))

    if filter_LD:
        valid_idx1 = (refs[0].sum(dim=0) - refs[2].sum(dim=0)).abs() >= min_freq_difference
    else:
        valid_idx1 = torch.ones((file_to_length[file_num],), dtype=torch.bool)

    positions = positions[valid_idx1]

    valid_idx2 = torch.ones(positions.shape, dtype=torch.bool)
    if filter_freq:
        last_morgan = -1
        for i in range(len(positions)):
            if positions[i]*morgans / base_pairs - last_morgan < morgan_cutoff:
                valid_idx2[i] = 0
            else:
                last_morgan = positions[i]*morgans / base_pairs

    positions = positions[valid_idx2]

    valid_idxs = torch.arange(file_to_length[file_num], dtype=torch.long)
    valid_idxs = valid_idxs[valid_idx1][valid_idx2]

    X = X[...,valid_idxs]
    y = y[...,valid_idxs]
    refs = refs[...,valid_idxs]

    idxs_out = None
    if idx_predict is not None:
        idxs_out = [(valid_idxs - idx).abs().argmin().item() for idx in idx_predict]


    return X, y, refs, positions, params, idxs_out
