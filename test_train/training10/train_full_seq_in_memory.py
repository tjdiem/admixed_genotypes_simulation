from globals import *
from processing import *
from training_functions import preprocess_batch
from kmodels2 import KNet4, TestNet
from kmodels_positional import KNet_positional
import sys

if mixed_precision:
    from torch.cuda.amp import autocast, GradScaler
    scaler = GradScaler()
    training_context = autocast
else:
    from contextlib import nullcontext
    training_context = nullcontext

save_inputs = False
chunk_size = 1000

save_file = sys.argv[1] if sys.argv[1].endswith(".pth") else sys.argv[1] + ".pth"

GetMemory()

if save_inputs:
    print("Loading and saving arrays...")

    panel_data = [convert_panel(panel_dir + "panel_" + str(i)) for i in range(num_files)]
    X = [x for x, _ in panel_data]
    positions = [pos for _, pos in panel_data]

    for file_num, X_example in enumerate(X):
        X_example = torch.tensor(X_example)
        for chunk_start in range(0, X_example.shape[-1], chunk_size):
            chunk_end = min(X_example.shape[-1], chunk_start + chunk_size)
            torch.save(X_example[:, chunk_start:chunk_end], f"saved_inputs/X_file{file_num}_chunk{chunk_start}.pt")

    for file_num, positions_example in enumerate(positions):
        positions_example = torch.tensor(positions_example)
        for chunk_start in range(0, positions_example.shape[-1], chunk_size):
            chunk_end = min(positions_example.shape[-1], chunk_start + chunk_size)
            torch.save(positions_example[chunk_start:chunk_end], f"saved_inputs/positions_file{file_num}_chunk{chunk_start}.pt")


    y = [convert_split(split_dir + "split_" + str(i), positions[i]) for i in range(num_files)]
    for file_num, y_example in enumerate(y):
        y_example = torch.tensor(y_example)
        for chunk_start in range(0, y_example.shape[-1], chunk_size):
            chunk_end = min(y_example.shape[-1], chunk_start + chunk_size)
            torch.save(y_example[::2, chunk_start:chunk_end] + y_example[1::2, chunk_start:chunk_end], f"saved_inputs/y_file{file_num}_chunk{chunk_start}.pt")

    panel_template_data = [convert_panel_template(panel_template_dir + "panel_template_" + str(i)) for i in range(num_files)]
    refA = [a for a, _, _ in panel_template_data]
    refB = [b for _, b, _ in panel_template_data]
    for file_num, (refa, refb) in enumerate(zip(refA, refB)):
        refa = torch.tensor(refa)
        refb = torch.tensor(refb)
        refs_example = torch.zeros((num_classes, n_ind_pan_model, refa.shape[-1]))
        refs_example[0] = refa[:2 * n_ind_pan // 6 * 2:2] + refa[1:2 * n_ind_pan // 6 * 2:2]
        refs_example[2] = refb[:2 * n_ind_pan // 6 * 2:2] + refb[1:2 * n_ind_pan // 6 * 2:2]
        refs_example[1] = refa[-(n_ind_pan // 6 * 2):] + refb[-(n_ind_pan // 6 * 2):]
        for chunk_start in range(0, refs_example.shape[-1], chunk_size):
            chunk_end = min(refs_example.shape[-1], chunk_start + chunk_size)
            torch.save(refs_example[:, :, chunk_start:chunk_end], f"saved_inputs/refs_file{file_num}_chunk{chunk_start}.pt")

    lengths = []
    for i in range(num_files):
        with open(panel_dir + "panel_" + str(i), "r") as f:
            length = sum(1 for line in f)
        lengths.append(length)

    with open("saved_inputs/file_to_length.txt", "w") as f:
        f.write(str(lengths))

    exit()

with open("saved_inputs/file_to_length.txt", "r") as f:
    file_to_length = eval(f.read())

# Scramble data
torch.manual_seed(random_seed)
random.seed(random_seed)
files = list(range(num_files))
random.shuffle(files)

# Split data
ind = int(train_prop * num_files)
files_train, files_test = files[:ind], files[ind:]
num_files_split = {"train": ind, "test": num_files - ind}

GetMemory()
GetTime()

# lr variables
lr = lr_start
lr_factor = (lr_end/lr_start)**(1/(num_epochs - 1))

# Define network
model = eval(model_name)()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

# Load pretrained transformer head
# pretrained_dict = torch.load('../training6/pretrained1.pth')
# model.load_state_dict(pretrained_dict, strict=False)

# for param in model.block1.parameters():
#     param.requires_grad = False

############## Augment test data
# Shuffle in sampling dimension
# idx = torch.randperm(num_chrom)
# X_test = X_test[:,:,idx]

# Randomly muliply each example by 1 or -1
# rand = torch.randint(0,2,size=(X_test.shape[0],1)) * 2 - 1
# X_test *= rand
#############

# Define functions for getting random batch and calculating loss
def get_batch(split, num_samples):

    files_sampled = files_train if split == "train" else files_test
    random.shuffle(files_sampled)
    files_sampled = files_sampled[:num_samples]

    X_sampled = []
    y_sampled = []
    refs_sampled = []
    positions_sampled = []
    params_sampled = []

    for file_num in files_sampled:
        max_input_length = 5 * input_size # hardcoded for now!!!!!!!!!
        input_length = random.randint(input_size, min(max_input_length, file_to_length[file_num])) 
        input_idx = random.sample(list(range(input_length)), input_size)
        input_idx = torch.tensor(input_idx)
        input_idx, _ = torch.sort(input_idx)
        input_length = input_idx[-1].item() - input_idx[0].item()
        start_idx = random.randint(0, file_to_length[file_num] - input_length - 1)
        end_idx = start_idx + input_length
        chunk_start = start_idx // chunk_size * chunk_size 
        chunk_end = end_idx // chunk_size * chunk_size
        chunks = [chunk for chunk in range(chunk_start, chunk_end + chunk_size, chunk_size)]
        input_idx = input_idx + start_idx - chunk_start - input_idx[0].item()

        X_chunk = torch.cat([torch.load(f"saved_inputs/X_file{file_num}_chunk{chunk}.pt") for chunk in chunks], dim=-1)
        X_example = X_chunk[..., input_idx]
        X_sampled.append(X_example)

        y_chunk = torch.cat([torch.load(f"saved_inputs/y_file{file_num}_chunk{chunk}.pt") for chunk in chunks], dim=-1)
        y_example = y_chunk[..., input_idx]
        y_sampled.append(y_example)
        
        refs_chunk = torch.cat([torch.load(f"saved_inputs/refs_file{file_num}_chunk{chunk}.pt") for chunk in chunks], dim=-1)
        refs_example = refs_chunk[..., input_idx]
        refs_sampled.append(refs_example)

        positions_chunk = torch.cat([torch.load(f"saved_inputs/positions_file{file_num}_chunk{chunk}.pt") for chunk in chunks], dim=-1)
        positions_example = positions_chunk[..., input_idx]
        positions_sampled.append(positions_example)

    X_sampled = torch.stack(X_sampled)
    y_sampled = torch.stack(y_sampled)
    refs_sampled = torch.stack(refs_sampled)
    positions_sampled = torch.stack(positions_sampled) / num_bp
    params_sampled = torch.tensor([convert_parameter(parameters_dir + "parameter_" + str(file_num)) for file_num in files_sampled])

    return X_sampled.to(device), y_sampled.to(device), refs_sampled.to(device), positions_sampled.to(device), params_sampled.to(device)

@torch.no_grad()
def estimate_loss(num_samples):
    model.eval()
    for split in ["train", "test"]:
        
        # X, y, refs, positions, params = get_batch(split, num_samples) #inside for loop
        y_pred = torch.zeros((num_samples, num_classes))
        y_true = torch.zeros((num_samples,)).long().to(device)
        params_tested = torch.zeros((num_samples, 6))

        for iter in range(ceil(num_samples / num_files_split[split])): #it's possible to make this slightly more efficient
            for istart in range(iter * num_files_split[split], min((iter + 1) * num_files_split[split], num_samples), batch_size):
               
                iend = min(istart + batch_size, (iter + 1) * num_files_split[split], num_samples)

                X_batch, y_batch, refs_batch, positions_batch, params_batch = get_batch(split, iend - istart)

                X_batch, y_batch, refs_batch, labels_batch = preprocess_batch(X_batch, y_batch, refs_batch)
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
    return loss

if GPU_available:
    print("GPU is available.")
    model = model.to("cuda")
    criterion = criterion.to("cuda")

else:
    print("No GPU available. Running on CPU.")

if checkpointing:
    from torch.utils.checkpoint import checkpoint

GetMemory()

#Training loop
best_loss = 1
model.train()
for epoch in range(num_epochs):
    if epoch == 0 or (epoch + 1) % eval_interval == 0:
        print("-----------------------------------------------------------------")
        print(f"Started training on epoch {epoch + 1} of {num_epochs}, learning rate {lr:0.7f}")
        GetTime()

    # Scramble data
    random.shuffle(files_train)

    for istart in range(0, len(files_train), batch_size):

        iend = min(istart + batch_size, len(files_train))

        X_batch, y_batch, refs_batch, positions_batch, params_batch = get_batch("train", iend - istart)

        X_batch, y_batch, refs_batch, labels_batch = preprocess_batch(X_batch, y_batch, refs_batch)
        labels_batch[labels_batch == -1] = 0

        optimizer.zero_grad()

        with training_context():

            try:
                y_pred = model(X_batch, refs_batch, labels_batch, positions_batch, params_batch)
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                y_pred = model(X_batch, refs_batch, labels_batch, positions_batch, params_batch)

            loss = criterion(y_pred, y_batch.to(device))

        if mixed_precision:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        else:
            loss.backward()
            optimizer.step()

    lr *= lr_factor
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    if (epoch + 1) % eval_interval == 0:

        loss = estimate_loss(num_estimate)

        if epoch == 1299:
            exit()

        if save_file.lower() != "none.pth" and loss < best_loss:
            best_loss = loss
            print("SAVING MODEL")
            torch.save(model.state_dict(), save_file)
