from globals import *
from processing import *
from kmodels2 import get_ref_sim
from training_functions import preprocess_batch


class Autoencoder(nn.Module):
    def __init__(self):
        super().__init__()

        hidden0 = n_embd_model
        hidden1 = 250
        
        # 250
        hidden0d = 2000
        hidden1d = 20000
        # ~100_000

        self.encoder0 = nn.Linear(n_embd_model, hidden0)

        self.encoder1 = nn.Linear(input_size * hidden0, hidden1)

        self.decoder0 = nn.Linear(hidden1, hidden0d)
        self.decoder1 = nn.Linear(hidden0d, hidden1d)
        self.decoder2 = nn.Linear(hidden1d, input_size * n_embd_model)

        self.relu = nn.ReLU()


    def forward(self, ref_sim):

        ref_sim = self.encoder0(ref_sim)

        ref_sim = ref_sim.reshape(*ref_sim.shape[:3], -1) # batch, num_classes, n_ind_max, input_size * hidden0
        ref_sim = self.encoder1(ref_sim)

        ref_sim = self.decoder0(ref_sim)
        ref_sim = self.relu(ref_sim)
        ref_sim = self.decoder1(ref_sim)
        ref_sim = self.relu(ref_sim)
        ref_sim = self.decoder2(ref_sim)

        ref_sim = ref_sim.reshape(*ref_sim.shape[:3], input_size, n_embd_model)

        return ref_sim

num_files = 367
batch_size = 16
input_size = 3001
num_estimate = 1000
lr_start = 3e-4
lr_end = lr_start / 1000
num_epochs = 2500
mixed_precision = False
checkpointing = False
time_counts = [0.0, 0.0, 0.0, 0.0]
chunk_size = 1000
save_file = sys.argv[1] if sys.argv[1].endswith(".pth") else sys.argv[1] + ".pth"

if mixed_precision:
    from torch.cuda.amp import autocast, GradScaler
    scaler = GradScaler()
    training_context = autocast
else:
    from contextlib import nullcontext
    training_context = nullcontext


# What conditions to include in test and train sets
pretrained_model = None #"../training10/diverse_13_2.pth"

max_input_length_splits = {"train": input_size, "test": input_size}

num_generations_valid = {"train": lambda x: True, "test": lambda x: True}

admixture_proportion_valid = {"train": lambda x: True, "test": lambda x: True}

preprocess_probs = {
    "prob0" : {"train": 0.5, "test": 0.5},
    "prob1" : {"train": 1.0, "test": 1.0},
    "prob2a" : {"train": 0.6, "test": 0.6},
    "prob2b" : {"train": 0.2, "test": 0.2},
    "prob2c" : {"train": 0.2, "test": 0.2},
    "prob3" : {"train": 1.0, "test": 1.0}
}

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
model = Autoencoder()
criterion = nn.MSELoss()
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

        loss = 0
        for iter in range(ceil(num_samples / num_files_split[split])): #it's possible to make this slightly more efficient
            for istart in range(iter * num_files_split[split], min((iter + 1) * num_files_split[split], num_samples), batch_size):
               
                t = time.time()
                files_sampled, idx_sampled = (files_train_eval, idx_train_eval) if split == "train" else (files_test_eval, idx_test_eval)
                iend = min(istart + batch_size, (iter + 1) * num_files_split[split], num_samples)

                X_batch, y_batch, refs_batch, positions_batch, params_batch = get_batch(split, [files_sampled[i % len(files_sampled)] for i in range(istart, iend)], idxs=[idx_sampled[i] for i in range(istart, iend)])

                X_batch, y_batch, refs_batch, labels_batch = preprocess_batch(X_batch, y_batch, refs_batch, prob_args=(preprocess_probs, split))
                labels_batch[labels_batch == -1] = 0

                params_tested[istart:iend] = params_batch
                ref_sim = get_ref_sim(X_batch, refs_batch, labels_batch, positions_batch, params_batch)
                print("a", time.time() - t)
                y_pred = model(ref_sim)
                # y_true[istart:iend] = y_batch

                print("b", time.time() - t)
                y_pred = y_pred.to(device)

                print(y_pred.shape)
                print(X_batch.shape)
                
                print("c", time.time() - t)
                loss += criterion(y_pred, ref_sim.to(device)).item()
                print("d", time.time() - t)
        # predictions = y_pred.argmax(dim=-1)
        # accuracy = (predictions == y_true).sum().item() / y_true.shape[0]

        print(f"Loss {split}: {loss:0.5f}")


                
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
# print("Initial assessment")
best_loss = estimate_loss(num_estimate)

time_counts = [0.0, 0.0, 0.0, 0.0]
# profiler.start()
#Training loop
model.train()
for epoch in range(num_epochs):
    if epoch == 0 or (epoch + 1) % eval_interval == 0:
        print("-----------------------------------------------------------------")
        print(f"Started training on epoch {epoch + 1} of {num_epochs}, learning rate {lr:0.7f}")
        GetTime()

    # Scramble data
    random.shuffle(files_train)
    for istart in range(0, len(files_train), batch_size):

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

            loss = criterion(y_pred, ref_sim.to(device))

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



        