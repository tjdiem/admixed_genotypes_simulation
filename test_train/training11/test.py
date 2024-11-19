from processing import *
from training_functions import *
from kmodels2 import KNet4
import torch
import random
import gzip

pretrained_model = "full_seq_3001.pth"

max_input_length_splits = {"train": 2 * input_size, "test": 2 * input_size}

num_generations_valid = {"train": lambda x: True, "test": lambda x: True}

admixture_proportion_valid = {"train": lambda x: True, "test": lambda x: True}

preprocess_probs = {
    "prob0" : {"train": 1.0, "test": 1.0},
    "prob1" : {"train": 0.5, "test": 0.5},
    "prob2a" : {"train": 0.0, "test": 0.0},
    "prob2b" : {"train": 0.0, "test": 0.0},
    "prob2c" : {"train": 0.0, "test": 0.0},
    "prob3" : {"train": 1.0, "test": 1.0}
}

num_files = 367
num_estimate = 1000
batch_size=16

save_inputs = False
chunk_size = 1000
in_memory = False
compressed = False

with open("saved_inputs/file_to_length.txt", "r") as f:
    file_to_length = eval(f.read())



# Scramble data
torch.manual_seed(random_seed)
random.seed(random_seed)
files = list(range(num_files))
# random.shuffle(files)

def is_valid_file(file_num, split):
    params_example = torch.tensor(convert_parameter(parameters_dir + "parameter_" + str(file_num)))
    return num_generations_valid[split](params_example[1].item()) and admixture_proportion_valid[split](params_example[0].item())


# Split data
ind = 100 #int(train_prop * num_files)
files_test, files_train = files[:ind], files[ind:]

is_valid_test = [is_valid_file(file, "test") for file in files_test]
is_valid_train = [is_valid_file(file, "train") for file in files_train]

files_train = [files_train[i] for i in range(len(files_train)) if is_valid_train[i]]
files_test = [files_test[i] for i in range(len(files_test)) if is_valid_test[i]]

files_train_eval = files_train.copy()
files_test_eval = files_test.copy()

num_files_split = {"train": len(files_train), "test": len(files_test)}
print("Num files: ", num_files_split)

with open("saved_inputs/idx_eval.txt", "r") as f:
    idx_eval = eval(f.read())
    idx_test_eval = idx_eval["test"]
    idx_train_eval = idx_eval["train"]

idx_test_eval = [idx for i, idx in enumerate(idx_test_eval) if is_valid_test[i % len(is_valid_test)]]

model = eval(model_name)()
model = model.to(device)
criterion = nn.CrossEntropyLoss()
if pretrained_model is not None:
    model.load_state_dict(torch.load(pretrained_model, map_location=torch.device(device)))


# Define functions for getting random batch and calculating loss
def get_batch(split, file_nums, idxs=None):

    X_sampled = []
    y_sampled = []
    refs_sampled = []
    positions_sampled = []
    params_sampled = []

    for i in range(len(file_nums)):

        file_num = file_nums[i]
        params_example = torch.tensor(convert_parameter(parameters_dir + "parameter_" + str(file_num)))

        params_sampled.append(params_example)
        
        max_input_length = max_input_length_splits[split]
        input_length = random.randint(input_size, min(max_input_length, file_to_length[file_num])) 
        input_idx_relative = random.sample(list(range(input_length)), input_size)
        input_idx_relative = torch.tensor(input_idx_relative)
        input_idx_relative, _ = torch.sort(input_idx_relative)
        middle_idx_relative = input_idx_relative[input_size // 2].item()

        middle_idx = random.randint(0, file_to_length[file_num] - 1) if idxs is None else idxs[i]
        input_idx = input_idx_relative + middle_idx - middle_idx_relative

        input_idx_valid = ((input_idx >= 0) & (input_idx < file_to_length[file_num]))
        assert input_idx_valid[input_size // 2].item()

        idx_valid = input_idx[input_idx_valid]
        start_idx = idx_valid[0].item()
        end_idx = idx_valid[-1].item()

        start_chunk = start_idx // chunk_size * chunk_size
        end_chunk = end_idx // chunk_size * chunk_size
        chunks = list(range(start_chunk, end_chunk + chunk_size, chunk_size))
        
        X_example = torch.full((n_ind_adm_end - n_ind_adm_start, input_size,), -1)
        y_example = torch.full((n_ind_adm_end - n_ind_adm_start, input_size,), -1)
        refs_example = torch.full((num_classes, n_ind_pan_model, input_size,), -1).float()
        positions_example = torch.full((input_size,), -1)
        
        if compressed:
            X_chunk = []
            for chunk in chunks:
                with gzip.open(f"saved_inputs/X_file{file_num}_chunk{chunk}.pt.gz", "rb") as f:
                    X_chunk.append(torch.load(f))
            X_chunk = torch.cat(X_chunk, dim=-1)
            X_chunk = X_chunk[:, idx_valid - start_chunk]
            X_example[:, input_idx_valid] = X_chunk

            y_chunk = []
            for chunk in chunks:
                with gzip.open(f"saved_inputs/y_file{file_num}_chunk{chunk}.pt.gz", "rb") as f:
                    y_chunk.append(torch.load(f))                
            y_chunk = y_chunk[:, idx_valid - start_chunk]
            y_example[:, input_idx_valid] = y_chunk

            refs_chunk = []
            for chunk in chunks:
                with gzip.open(f"saved_inputs/refs_file{file_num}_chunk{chunk}.pt.gz", "rb") as f:
                    refs_chunk.append(torch.load(f))
            refs_chunk = refs_chunk[:, :, idx_valid - start_chunk]
            refs_example[:, :, input_idx_valid] = refs_chunk

            positions_chunk = []
            for chunk in chunks:
                with gzip.open(f"saved_inputs/positions_file{file_num}_chunk{chunk}.pt.gz", "rb") as f:
                    positions_chunk.append(torch.load(f))
            positions_chunk = positions_chunk[idx_valid - start_chunk]
            positions_example[input_idx_valid] = positions_chunk
        else:     
            X_chunk = torch.cat([torch.load(f"saved_inputs/X_file{file_num}_chunk{chunk}.pt") for chunk in chunks], dim=-1)
            X_chunk = X_chunk[:, idx_valid - start_chunk]
            X_example[:, input_idx_valid] = X_chunk.long()

            y_chunk = torch.cat([torch.load(f"saved_inputs/y_file{file_num}_chunk{chunk}.pt") for chunk in chunks], dim=-1)
            y_chunk = y_chunk[:, idx_valid - start_chunk]
            y_example[:, input_idx_valid] = y_chunk.long()

            refs_chunk = torch.cat([torch.load(f"saved_inputs/refs_file{file_num}_chunk{chunk}.pt") for chunk in chunks], dim=-1)
            refs_chunk = refs_chunk[:, :, idx_valid - start_chunk]
            refs_example[:, :, input_idx_valid] = refs_chunk.float()

            positions_chunk = torch.cat([torch.load(f"saved_inputs/positions_file{file_num}_chunk{chunk}.pt") for chunk in chunks], dim=-1)
            positions_chunk = positions_chunk[idx_valid - start_chunk]
            positions_example[input_idx_valid] = positions_chunk


            ### parallel execution

        X_sampled.append(X_example)
        y_sampled.append(y_example)
        refs_sampled.append(refs_example)
        positions_sampled.append(positions_example)

    X_sampled = torch.stack(X_sampled)
    y_sampled = torch.stack(y_sampled)
    refs_sampled = torch.stack(refs_sampled)
    positions_sampled = torch.stack(positions_sampled) / num_bp
    params_sampled = torch.stack(params_sampled)
    # params_sampled = torch.tensor([convert_parameter(parameters_dir + "parameter_" + str(file_num)) for file_num in files_sampled])

    return X_sampled.to(device), y_sampled.to(device), refs_sampled.to(device), positions_sampled.to(device), params_sampled.to(device)

@torch.no_grad()
def estimate_loss(num_samples):

    random_seed = random.getstate()
    torch_seed = torch.randint(0, 2**32, (1,)).item()

    model.eval()
    for split in ["test"]:

        random.seed(123)
        torch.manual_seed(123)
        
        # X, y, refs, positions, params = get_batch(split, num_samples) #inside for loop
        y_pred = torch.zeros((num_samples, num_classes))
        y_true = torch.zeros((num_samples,)).long().to(device)
        params_tested = torch.zeros((num_samples, 6))

        for iter in range(ceil(num_samples / num_files_split[split])): #it's possible to make this slightly more efficient
            for istart in range(iter * num_files_split[split], min((iter + 1) * num_files_split[split], num_samples), batch_size):
               
                files_sampled, idx_sampled = (files_train_eval, idx_train_eval) if split == "train" else (files_test_eval, idx_test_eval)
                iend = min(istart + batch_size, (iter + 1) * num_files_split[split], num_samples)

                X_batch, y_batch, refs_batch, positions_batch, params_batch = get_batch(split, [files_sampled[i % len(files_sampled)] for i in range(istart, iend)], idxs=[idx_sampled[i] for i in range(istart, iend)])

                X_batch, y_batch, refs_batch, labels_batch = preprocess_batch(X_batch, y_batch, refs_batch, prob_args=(preprocess_probs, split))
                labels_batch[labels_batch == -1] = 0

                params_tested[istart:iend] = params_batch
                y_pred[istart:iend] = model(X_batch, refs_batch, labels_batch, positions_batch, params_batch)
                y_true[istart:iend] = y_batch


        if GPU_available:
            y_pred = y_pred.to("cuda")
        
        loss = criterion(y_pred, y_true).item()
        predictions = y_pred.argmax(dim=-1)
        accuracy = (predictions == y_true).sum().item() / y_true.shape[0]

        print(f"Loss {split}: {loss:0.5f}, Accuracy: {accuracy:0.4f}")


        correct = (predictions == y_true)
        threshold_num_examples = 10
        splits0 = [2000, 3000, 4000, 5000, 6000, 7000, 8000]
        splits1 = [20000, 22000, 24000, 26000, 28000, 30000]
        grid = torch.zeros((len(splits0) - 1, len(splits1) - 1))
        for i in range(len(splits0) - 1):
            for j in range(len(splits1) - 1):
                valid_indices = ((params_tested[:,4] > splits0[i]) &
                                 (params_tested[:,4] < splits0[i + 1]) &
                                 (params_tested[:,3] > splits1[j]) &
                                 (params_tested[:,3] < splits1[j + 1]))
                
                num_examples = valid_indices.sum().item()
                if num_examples < threshold_num_examples:
                    grid[i,j] = -1
                else:
                    grid[i,j] = correct[valid_indices].sum() / num_examples
                
        print(grid)
    
    model.train()

    random.setstate(random_seed)
    torch.manual_seed(torch_seed)
    return loss

estimate_loss(num_estimate)

# for file_num in files_test:
#     X = torch.cat([torch.load(f"saved_inputs/X_file{file_num}_chunk{chunk}.pt") for chunk in range(0, file_to_length[file_num] // chunk_size * chunk_size + chunk_size, chunk_size)], dim=-1).long()
#     y = torch.cat([torch.load(f"saved_inputs/y_file{file_num}_chunk{chunk}.pt") for chunk in range(0, file_to_length[file_num] // chunk_size * chunk_size + chunk_size, chunk_size)], dim=-1).long()
#     refs = torch.cat([torch.load(f"saved_inputs/refs_file{file_num}_chunk{chunk}.pt") for chunk in range(0, file_to_length[file_num] // chunk_size * chunk_size + chunk_size, chunk_size)], dim=-1).float()
#     positions = torch.cat([torch.load(f"saved_inputs/positions_file{file_num}_chunk{chunk}.pt") for chunk in range(0, file_to_length[file_num] // chunk_size * chunk_size + chunk_size, chunk_size)], dim=-1)
#     params = torch.tensor(convert_parameter(parameters_dir + "parameter_" + str(file_num)))

