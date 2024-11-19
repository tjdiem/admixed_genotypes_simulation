from globals import *
from processing import *
from kmodels2_testing19 import KNet4, get_ref_sim
from training_functions import *
import torch.cuda.profiler as profiler
import sys
import gzip
import os 
import shutil
import concurrent.futures

# What conditions to include in test and train sets
pretrained_model = "diverse_3001_19_4.pth"

# max_input_length_splits = {"train": input_size, "test": input_size}

# num_generations_valid = {"train": lambda x: True, "test": lambda x: True}

# admixture_proportion_valid = {"train": lambda x: True, "test": lambda x: True}

# preprocess_probs = {
#     "prob0" : {"train": 0.5, "test": 0.5},
#     "prob1" : {"train": 1.0, "test": 1.0},
#     "prob2a" : {"train": 0.6, "test": 0.6},
#     "prob2b" : {"train": 0.2, "test": 0.2},
#     "prob2c" : {"train": 0.2, "test": 0.2},
#     "prob3" : {"train": 1.0, "test": 1.0}
# }

# max_input_length_splits = {"train": 5 * input_size, "test": 2 * input_size}

# num_generations_valid = {"train": lambda x: True, "test": lambda x: True}

# admixture_proportion_valid = {"train": lambda x: True, "test": lambda x: True}

# preprocess_probs = {
#     "prob0" : {"train": 0.5, "test": 1.0},
#     "prob1" : {"train": 0.5, "test": 0.5},
#     "prob2a" : {"train": 0.6, "test": 0.0},
#     "prob2b" : {"train": 0.2, "test": 0.0},
#     "prob2c" : {"train": 0.2, "test": 0.0},
#     "prob3" : {"train": 1.0, "test": 1.0}
# }

max_input_length_splits = {"train": input_size, "test": input_size}

num_generations_valid = {"train": lambda x: True, "test": lambda x: True}

admixture_proportion_valid = {"train": lambda x: True, "test": lambda x: True}

preprocess_probs = {
    "prob0" : {"train": 1.0, "test": 1.0},
    "prob1" : {"train": 0.5, "test": 0.5},
    "prob2a" : {"train": 0.0, "test": 0.0},
    "prob2b" : {"train": 0.0, "test": 0.0},
    "prob2c" : {"train": 0.0, "test": 0.0},
    "prob2d" : {"train": 0.075, "test": 0.075},
    "prob2e" : {"train": 0.075, "test": 0.075},
    "prob3" : {"train": 1.0, "test": 1.0}
}

# max_input_length_splits = {"train": input_size, "test": input_size}

# num_generations_valid = {"train": lambda x: True, "test": lambda x: True}

# admixture_proportion_valid = {"train": lambda x: True, "test": lambda x: True}

# preprocess_probs = {
#     "prob0" : {"train": 1.0, "test": 1.0},
#     "prob1" : {"train": 0.5, "test": 0.5},
#     "prob2a" : {"train": 0.0, "test": 0.0},
#     "prob2b" : {"train": 0.0, "test": 0.0},
#     "prob2c" : {"train": 0.0, "test": 0.0},
#     "prob3" : {"train": 1.0, "test": 1.0}
# }


time_counts = [0.0, 0.0, 0.0, 0.0]

if mixed_precision:
    from torch.cuda.amp import autocast, GradScaler
    scaler = GradScaler()
    training_context = autocast
else:
    from contextlib import nullcontext
    training_context = nullcontext

save_inputs = False
chunk_size = 1000
in_memory = False
compressed = False

save_file = sys.argv[1] if sys.argv[1].endswith(".pth") else sys.argv[1] + ".pth"

GetMemory()

def gzip_file(file):
    if compressed:
        with open(file, "rb") as f_in:
            with gzip.open(file + ".gz", "wb") as f_out:
                shutil.copyfileobj(f_in, f_out)

        os.remove(file)


if save_inputs:
    print("Loading and saving arrays...")

    proc = int(sys.argv[2])
    num_proc = int(sys.argv[3])
    for file_num in range(num_files):
        if file_num % num_proc != proc:
            continue
        x, pos = convert_panel(panel_dir + "panel_" + str(file_num))
        y = convert_split(split_dir + "split_" + str(file_num), pos)
        x = torch.tensor(x)
        pos = torch.tensor(pos)
        y = torch.tensor(y)
        for chunk_start in range(0, x.shape[-1], chunk_size):
            chunk_end = min(x.shape[-1], chunk_start + chunk_size)
            torch.save(x[:, chunk_start:chunk_end].clone().to(torch.int8), f"saved_inputs/X_file{file_num}_chunk{chunk_start}.pt")
            gzip_file(f"saved_inputs/X_file{file_num}_chunk{chunk_start}.pt")
        for chunk_start in range(0, pos.shape[-1], chunk_size):
            chunk_end = min(pos.shape[-1], chunk_start + chunk_size)
            torch.save(pos[chunk_start:chunk_end].clone().long(), f"saved_inputs/positions_file{file_num}_chunk{chunk_start}.pt")
            gzip_file(f"saved_inputs/positions_file{file_num}_chunk{chunk_start}.pt")
        for chunk_start in range(0, y.shape[-1], chunk_size):
            chunk_end = min(y.shape[-1], chunk_start + chunk_size)
            torch.save(y[::2, chunk_start:chunk_end].clone().to(torch.int8) + y[1::2, chunk_start:chunk_end].clone().int(), f"saved_inputs/y_file{file_num}_chunk{chunk_start}.pt")
            gzip_file(f"saved_inputs/y_file{file_num}_chunk{chunk_start}.pt")

    # panel_data = [convert_panel(panel_dir + "panel_" + str(i)) for i in range(num_files)]
    # X = [x for x, _ in panel_data]
    # positions = [pos for _, pos in panel_data]

    # for file_num, X_example in enumerate(X):
    #     X_example = torch.tensor(X_example)
    #     for chunk_start in range(0, X_example.shape[-1], chunk_size):
    #         chunk_end = min(X_example.shape[-1], chunk_start + chunk_size)
    #         torch.save(X_example[:, chunk_start:chunk_end], f"saved_inputs/X_file{file_num}_chunk{chunk_start}.pt")

    # for file_num, positions_example in enumerate(positions):
    #     positions_example = torch.tensor(positions_example)
    #     for chunk_start in range(0, positions_example.shape[-1], chunk_size):
    #         chunk_end = min(positions_example.shape[-1], chunk_start + chunk_size)
    #         torch.save(positions_example[chunk_start:chunk_end], f"saved_inputs/positions_file{file_num}_chunk{chunk_start}.pt")


    # y = [convert_split(split_dir + "split_" + str(i), positions[i]) for i in range(num_files)]
    # for file_num, y_example in enumerate(y):
    #     y_example = torch.tensor(y_example)
    #     for chunk_start in range(0, y_example.shape[-1], chunk_size):
    #         chunk_end = min(y_example.shape[-1], chunk_start + chunk_size)
    #         torch.save(y_example[::2, chunk_start:chunk_end] + y_example[1::2, chunk_start:chunk_end], f"saved_inputs/y_file{file_num}_chunk{chunk_start}.pt")

    # panel_template_data = [convert_panel_template(panel_template_dir + "panel_template_" + str(i)) for i in range(num_files)]
    # refA = [a for a, _, _ in panel_template_data]
    # refB = [b for _, b, _ in panel_template_data]
    # for file_num, (refa, refb) in enumerate(zip(refA, refB)):
    #     if file_num % num_proc != proc:
    #         continue
    #     refa = torch.tensor(refa)
    #     refb = torch.tensor(refb)
    #     refs_example = torch.zeros((num_classes, n_ind_pan_model, refa.shape[-1]))
    #     refs_example[0] = refa[:2 * n_ind_pan // 6 * 2:2] + refa[1:2 * n_ind_pan // 6 * 2:2]
    #     refs_example[2] = refb[:2 * n_ind_pan // 6 * 2:2] + refb[1:2 * n_ind_pan // 6 * 2:2]
    #     refs_example[1] = refa[-(n_ind_pan // 6 * 2):] + refb[-(n_ind_pan // 6 * 2):]
    #     for chunk_start in range(0, refs_example.shape[-1], chunk_size):
    #         chunk_end = min(refs_example.shape[-1], chunk_start + chunk_size)
    #         torch.save(refs_example[:, :, chunk_start:chunk_end].clone().to(torch.int8), f"saved_inputs/refs_file{file_num}_chunk{chunk_start}.pt")
    #         gzip_file(f"saved_inputs/refs_file{file_num}_chunk{chunk_start}.pt")
            
    for file_num in range(num_files):
        refa, refb, _ = convert_panel_template(panel_template_dir + "panel_template_" + str(file_num))
        if file_num % num_proc != proc:
            continue
        refa = torch.tensor(refa)
        refb = torch.tensor(refb)
        refs_example = torch.zeros((num_classes, n_ind_pan_model, refa.shape[-1]))
        refs_example[0] = refa[:2 * n_ind_pan // 6 * 2:2] + refa[1:2 * n_ind_pan // 6 * 2:2]
        refs_example[2] = refb[:2 * n_ind_pan // 6 * 2:2] + refb[1:2 * n_ind_pan // 6 * 2:2]
        refs_example[1] = refa[-(n_ind_pan // 6 * 2):] + refb[-(n_ind_pan // 6 * 2):]
        for chunk_start in range(0, refs_example.shape[-1], chunk_size):
            chunk_end = min(refs_example.shape[-1], chunk_start + chunk_size)
            torch.save(refs_example[:, :, chunk_start:chunk_end].clone().to(torch.int8), f"saved_inputs/refs_file{file_num}_chunk{chunk_start}.pt")
            gzip_file(f"saved_inputs/refs_file{file_num}_chunk{chunk_start}.pt")


    if proc == 0:
        lengths = []
        for i in range(num_files):
            try:
                with open(panel_dir + "panel_" + str(i), "r") as f:
                    length = sum(1 for _ in f)
            except FileNotFoundError:
                with tarfile.open(panel_dir + "panel_" + str(i) + ".tar.xz", "r:xz") as tar:
                    file = tar.extractfile(tar.getnames()[0]) #"panel_" + str(i))
                    length = sum(1 for _ in file)
            lengths.append(length)

        with open("saved_inputs/file_to_length.txt", "w") as f:
            f.write(str(lengths))

    exit()

with open("saved_inputs/file_to_length.txt", "r") as f:
    file_to_length = eval(f.read())

if in_memory:
    X = [torch.cat([torch.load(f"saved_inputs/X_file{file_num}_chunk{i}.pt") for i in range(0, file_to_length[file_num], chunk_size)], dim=-1) for file_num in range(num_files)]
    y = [torch.cat([torch.load(f"saved_inputs/y_file{file_num}_chunk{i}.pt") for i in range(0, file_to_length[file_num], chunk_size)], dim=-1) for file_num in range(num_files)]
    refs = [torch.cat([torch.load(f"saved_inputs/refs_file{file_num}_chunk{i}.pt") for i in range(0, file_to_length[file_num], chunk_size)], dim=-1) for file_num in range(num_files)]
    positions = [torch.cat([torch.load(f"saved_inputs/positions_file{file_num}_chunk{i}.pt") for i in range(0, file_to_length[file_num], chunk_size)], dim=-1) for file_num in range(num_files)]
    
    print("Finished loading data")
    GetTime()
    GetMemory()

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

files_train = [file for file in files_train if is_valid_file(file, "train")]
files_test = [file for file in files_test if is_valid_file(file, "test")]

num_files_split = {"train": len(files_train), "test": len(files_test)}
print("Num files: ", num_files_split)


files_train_eval = files_train.copy()
files_test_eval = files_test.copy()
idx_train_eval = [random.randint(0, file_to_length[files_train_eval[i % num_files_split["train"]]] - 1) for i in range(num_estimate)]
idx_test_eval = [random.randint(0, file_to_length[files_test_eval[i % num_files_split["test"]]] - 1) for i in range(num_estimate)]

with open("saved_inputs/idx_eval.txt", "w") as f:
    f.write(str({"test": idx_test_eval, "train": idx_train_eval}))


GetMemory()
GetTime()

# lr variables
lr = lr_start
lr_factor = (lr_end/lr_start)**(1/(num_epochs - 1))

# Define network
model = eval(model_name)()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
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
        
        if in_memory:
            X_example[:, input_idx_valid] = X[file_num][:, idx_valid]
            y_example[:, input_idx_valid] = y[file_num][:, idx_valid]
            refs_example[:, :, input_idx_valid] = refs[file_num][:, :, idx_valid]
            positions_example[input_idx_valid] = positions[file_num][idx_valid]

        else:
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
    for split in ["train", "test"]:

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
                ref_sim = get_ref_sim(X_batch, refs_batch, labels_batch, positions_batch, params_batch)
                y_pred[istart:iend] = model(ref_sim)
                y_true[istart:iend] = y_batch


        y_pred = y_pred.to(device)
        
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
                
        # print(grid)
    
    model.train()

    random.setstate(random_seed)
    torch.manual_seed(torch_seed)
    return loss

if device == "cuda":
    print("GPU is available.")
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    model = model.to(device)
    criterion = criterion.to(device)

else:
    print("No GPU available. Running on CPU.")

if checkpointing:
    from torch.utils.checkpoint import checkpoint

GetMemory()

# num_estimate = min(num_estimate, len(files_test))
# if pretrained_model is not None:
best_loss = 1
print("Initial assessment")
best_loss = estimate_loss(num_estimate)

# profiler.start()
# Training loop
model.train()
for epoch in range(num_epochs):
    
    if epoch == 0 or (epoch + 1) % eval_interval == 0:
        print("-----------------------------------------------------------------")
        print(f"Started training on epoch {epoch + 1} of {num_epochs}, learning rate {lr:0.7f}")
        GetTime()

    # Scramble data
    random.shuffle(files_train)
    for istart in range(0, len(files_train), batch_size):

        # torch.cuda.empty_cache()


        time_start0 = time.time()
        iend = min(istart + batch_size, len(files_train))

        X_batch, y_batch, refs_batch, positions_batch, params_batch = get_batch("train", [files_train[i] for i in range(istart, iend)])

        time_start1 = time.time()

        X_batch, y_batch, refs_batch, labels_batch = preprocess_batch(X_batch, y_batch, refs_batch, prob_args=(preprocess_probs, "train"))
        labels_batch[labels_batch == -1] = 0

        time_start2 = time.time()

        optimizer.zero_grad()

        with training_context():
            try:
                ref_sim = get_ref_sim(X_batch.clone(), refs_batch.clone(), labels_batch.clone(), positions_batch.clone(), params_batch.clone())
                y_pred = model(ref_sim)
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                ref_sim = get_ref_sim(X_batch, refs_batch, labels_batch, positions_batch, params_batch)
                y_pred = model(ref_sim)

            loss = criterion(y_pred, y_batch.to(device))

        time_start3 = time.time()

        if mixed_precision:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            # 16 * 3 * 16 * 501 * 5001

        else:
            # print(2)
            # GetCudaMemory()
            loss.backward()
            optimizer.step()

        time_start4 = time.time()
        # print(time_start4 - time_start0)

        time_counts[0] += time_start1 - time_start0 # get_batch function
        time_counts[1] += time_start2 - time_start1 # preprocess_batch function
        time_counts[2] += time_start3 - time_start2 # forward pass
        time_counts[3] += time_start4 - time_start3 # backward pass


    lr *= lr_factor
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    if (epoch + 1) % eval_interval == 0:

        print(time_counts)
        loss = estimate_loss(num_estimate)

        if save_file.lower() != "none.pth" and loss < best_loss:
            best_loss = loss
            print("SAVING MODEL")
            torch.save(model.state_dict(), save_file)


    # print(torch.cuda.memory_summary(device=None, abbreviated=False))



#[701.487804889679, 102.0738422870636, 499.4680688381195, 1.801103115081787]
