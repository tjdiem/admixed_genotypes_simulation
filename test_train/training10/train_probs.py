from globals import *
from processing import *
from training_functions import *
from kmodels2 import KNet4
import sys

if mixed_precision:
    from torch.cuda.amp import autocast, GradScaler
    scaler = GradScaler()
    training_context = autocast
else:
    from contextlib import nullcontext
    training_context = nullcontext

save_file = sys.argv[1] if sys.argv[1].endswith(".pth") else sys.argv[1] + ".pth"

GetMemory()
panel_data = [convert_panel(panel_dir + "panel_" + str(i)) for i in range(num_files)]
X = [x for x, _ in panel_data]
positions = [pos for _, pos in panel_data]
X = torch.tensor(X) # num_files, n_ind_adm, input_size

y = [convert_split(split_dir + "split_" + str(i), positions[i]) for i in range(num_files)]
y = torch.tensor(y) # num_files, 2 * n_ind_adm, input_size
y = y[:,::2] + y[:,1::2] # unphase ancestry labels # same shape as X

panel_template_data = [convert_panel_template(panel_template_dir + "panel_template_" + str(i)) for i in range(num_files)]
refA = [a for a, _, _ in panel_template_data]
refB = [b for _, b, _ in panel_template_data]
assert positions == [pos for _, _, pos in panel_template_data]

params = [convert_parameter(parameters_dir + "parameter_" + str(i)) for i in range(num_files)]
params = torch.tensor(params) # num_files, 6

refA = torch.tensor(refA) #num_files, n_ind_pan, input_size
refB = torch.tensor(refB) #num_files, n_ind_pan, input_size
refs = torch.zeros((num_files, num_classes, n_ind_pan_model, input_size))
refs[:,0] = refA[:, :2 * n_ind_pan // 6 * 2:2] + refA[:, 1:2 * n_ind_pan // 6 * 2:2]
refs[:,2] = refB[:, :2 * n_ind_pan // 6 * 2:2] + refB[:, 1:2 * n_ind_pan // 6 * 2:2]
refs[:,1] = refA[:, -(n_ind_pan // 6 * 2):] + refB[:, -(n_ind_pan // 6 * 2):]
positions = torch.tensor(positions)

# print(refs.shape)
# print(refs.max())
# print(refs.min())
# # print(((X == 1).sum(dim=-1) > 0).sum().item())

# print()
# for i in range(3):
#     print((refs == i).sum().item() / refs.numel())
# print()
# for i in range(3):
#     print((y == i).sum().item() / y.numel())
# print()
# for i in range(3):
#     print((X == i).sum().item() / X.numel())
# print()
# exit()

# print(X.shape)
# print(y.shape)
# print(refs.shape)
# print(positions.shape)
# exit()

GetMemory()

# Scramble data
torch.manual_seed(random_seed)
idx = torch.randperm(X.shape[0])
X = X[idx]
y = y[idx]
refs = refs[idx]
params = params[idx]

# Split data
ind = int(train_prop * X.shape[0])
X_train, X_test = X[:ind], X[ind:]
y_train, y_test = y[:ind], y[ind:]
refs_train, refs_test = refs[:ind], refs[ind:]
params_train, params_test = params[:ind], params[ind:]

GetMemory()
GetTime()

# lr variables
lr = lr_start
lr_factor = (lr_end/lr_start)**(1/(num_epochs - 1))

# Define network
model = KNet4()
model.load_state_dict(torch.load(start_model, map_location=torch.device(device)))
criterion = nn.KLDivLoss(reduction='batchmean')
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
    X, y, refs, params = (X_train, y_train, refs_train, params_train) if split == "train" else (X_test, y_test, refs_test, params_test)
    idx = torch.randperm(X.shape[0])
    idx = idx[:min(num_samples,len(idx))]
    X = X[idx]
    y = y[idx]
    refs = refs[idx]
    params = params[idx]
    return X.to(device), y.to(device), refs, params

@torch.no_grad()
def estimate_loss(num_samples):
    model.eval()
    for split in ["train", "test"]:
        for num_probs in [0, 2, 4]:  #hardcoded for now
            X, y, refs, params = get_batch(split, num_samples)
            y_pred = torch.zeros((num_samples, num_classes))
            y_true = torch.zeros((num_samples, num_classes)).to(device)
            params_tested = torch.zeros((num_samples, 6))
            for iter in range(ceil(num_samples / X.shape[0])): #it's possible to make this slightly more efficient
                for istart in range(iter * X.shape[0], min((iter + 1) * X.shape[0], num_samples), batch_size):
                
                    iend = min(istart + batch_size, (iter + 1) * X.shape[0], num_samples)

                    X_batch = X[istart % X.shape[0]:(iend - 1) % X.shape[0] + 1].to(device)
                    y_batch = y[istart % X.shape[0]:(iend - 1) % X.shape[0] + 1]
                    refs_batch = refs[istart % X.shape[0]:(iend - 1) % X.shape[0] + 1].clone().to(device)
                    params_tested[istart:iend] = params[istart % X.shape[0]:(iend - 1) % X.shape[0] + 1]

                    X_batch, y_batch, refs_batch, labels_batch = preprocess_batch(X_batch, y_batch, refs_batch)
                    labels_batch, y_batch = get_y_batch(X_batch, y_batch, refs_batch, labels_batch, model, num_probs=num_probs)

                    if labels_batch is None:
                        continue

                    y_pred[istart:iend] = model(X_batch, refs_batch, labels_batch)
                    y_true[istart:iend] = y_batch


            if GPU_available:
                y_pred = y_pred.to("cuda")
            
            valid_indices = y_true.sum(dim=-1) > 0
            y_pred = y_pred[valid_indices]
            y_true = y_true[valid_indices]

            y_pred_log = F.log_softmax(y_pred, dim=-1)
            y_pred = F.softmax(y_pred, dim=-1)
            loss = criterion(y_pred_log, y_true).item()

            str_out = f"Dataset: {split}, Num Probs: {num_probs}, Loss: {loss:0.12f}"

            if num_probs == 0:
                accuracy = (y_pred.argmax(dim=-1) == y_true.argmax(dim=-1)).sum().item() / y_true.shape[0]
                str_out += f", Accuracy: {accuracy:0.4f}"
            else:
                avg_dist = ((y_pred - y_true).abs().max(dim=-1)[0].mean())
                str_out += f", Avg Dist: {avg_dist:0.9f}"

            print(str_out)


        # correct = (predictions == y_true)
        # threshold_num_examples = 10
        # splits0 = [2000, 3000, 4000, 5000, 6000, 7000, 8000]
        # splits1 = [20000, 22000, 24000, 26000, 28000, 30000]
        # grid = torch.zeros((len(splits0) - 1, len(splits1) - 1))
        # for i in range(len(splits0) - 1):
        #     for j in range(len(splits1) - 1):
        #         valid_indices = ((params_tested[:,4] > splits0[i]) &
        #                          (params_tested[:,4] < splits0[i + 1]) &
        #                          (params_tested[:,3] > splits1[j]) &
        #                          (params_tested[:,3] < splits1[j + 1]))
                
        #         num_examples = valid_indices.sum().item()
        #         if num_examples < threshold_num_examples:
        #             grid[i,j] = -1
        #         else:
        #             grid[i,j] = correct[valid_indices].sum() / num_examples
                
        # print(grid)

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
    idx = torch.randperm(X_train.shape[0])
    X_train = X_train[idx]
    y_train = y_train[idx]
    refs_train = refs_train[idx]
    params_train = params_train[idx]

    for istart in range(0, X_train.shape[0], batch_size):

        iend = min(istart + batch_size, X_train.shape[0])

        y_batch = y_train[istart:iend]
        X_batch = X_train[istart:iend].to(device)
        refs_batch = refs_train[istart:iend].clone().to(device)

        X_batch, y_batch, refs_batch, labels_batch = preprocess_batch(X_batch, y_batch, refs_batch)

        labels_batch, y_batch = get_y_batch(X_batch, y_batch, refs_batch, labels_batch, model)

        optimizer.zero_grad()

        with training_context():

            try:
                y_pred = model(X_batch, refs_batch, labels_batch)
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                y_pred = model(X_batch, refs_batch, labels_batch)

            y_pred = F.log_softmax(y_pred, dim=-1)
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

        if save_file.lower() != "none.pth" and loss < best_loss:
            best_loss = loss
            print("SAVING MODEL")
            torch.save(model.state_dict(), save_file)
