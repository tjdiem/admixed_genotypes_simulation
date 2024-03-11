from globals import *
from processing import *
from models import *
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
Data = [convert_output_file(panel_dir + "panel_" + str(i), phase_dir + "phase_" + str(i)) for i in range(num_files)]

X = [xx for xx, _ in Data if xx is not None]
y = [yy for _, yy in Data if yy is not None]

# print([y[0][i][0] for i in range(len(y[0]))])

X = torch.tensor(X).float() - 1 #(num usable files, len_chrom, num_samples)
y = torch.tensor(y)

print(X.shape)
print(y.shape)

# print((y.max(axis=0)[0] - y.min(axis=0)[0]).sum().item())

# import matplotlib.pyplot as plt

# for i in range(20):
#     j = int(random.random() * input_size_processing)
#     k = int(random.random() * n_embd_processing)
#     q = y[:,j, k]
#     plt.hist(q)
#     plt.show()

# y = y.reshape(-1)
# plt.hist(y)
# plt.show()

# print(X.max(), X.min())
# print(y.max(), y.min())

# print(X.shape)
# print(y.shape)


GetMemory()

# Scramble data
torch.manual_seed(random_seed)
idx = torch.randperm(X.shape[0])
X = X[idx]
y = y[idx]

# X = X[:, :input_size, 101 - n_embd:]
# y = y[:, :input_size, 100 - n_embd:]

# Split data
ind = int(train_prop * X.shape[0])
X_train, X_test = X[:ind], X[ind:]
y_train, y_test = y[:ind], y[ind:]

GetMemory()
GetTime()

# lr variables
lr = lr_start
lr_factor = (lr_end/lr_start)**(1/(num_epochs - 1))

# Define network
model = TransformerModel2()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

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
    X, y = (X_train, y_train) if split == "train" else (X_test, y_test)
    idx = torch.randperm(X.shape[0])[:num_samples]
    X = X[idx]
    y = y[idx]
    if GPU_available:
        return X.to("cuda"), y.to("cuda")
    else:
        return X, y

@torch.no_grad()
def estimate_loss(num_samples):
    model.eval()
    for split in ["train", "test"]:
        X, y = get_batch(split, num_samples)
        y_pred = torch.zeros((num_samples, num_classes))
        y_true = torch.zeros((num_samples,)).long().to(device)

        for istart in range(0, num_samples, batch_size):
            iend = min(istart + batch_size, num_samples)
            random_indices1 = torch.randint(0, input_size, (iend - istart,))
            random_indices2 = torch.randint(0, n_embd_processing, (iend - istart,))

            X_batch = X[istart:iend]
            output_info = torch.zeros_like(X_batch).long()
        
            for k in range(iend - istart):  # should be parallelizable
                valid_y = torch.nonzero(y[istart:iend][k,:,random_indices2[k]] - 1)
                valid_index = valid_y[torch.randint(0, valid_y.shape[0], (1,)).item()]

                output_info[k,valid_index,random_indices2[k]] = y[istart:iend][k,valid_index,random_indices2[k]] - 1

            y_pred[istart:iend] = model(X_batch, random_indices1.to(device), random_indices2.to(device), output_info)

            y_true[istart:iend] = y[torch.arange(istart, iend),random_indices1, random_indices2]

        if GPU_available:
            y_pred = y_pred.to("cuda")

        print(y_pred, y_true)
        
        loss = criterion(y_pred, y_true).item()
        predictions = y_pred.argmax(dim=-1)
        accuracy = (predictions == y_true).sum().item() / num_samples

        print(f"Loss {split}: {loss:0.5f}, Accuracy: {accuracy:0.4f}")

    return loss


if GPU_available:
    print("GPU is available.")
    model = model.to("cuda")
    criterion = criterion.to("cuda")

else:
    print("No GPU available. Running on CPU.")

if checkpointing:
    from torch.utils.checkpoint import checkpoint
    model.forward = checkpoint(model.forward)

GetMemory()

#Training loop
best_loss = 1
model.train()
for epoch in range(num_epochs):
    print("-----------------------------------------------------------------")
    print(f"Started training on epoch {epoch + 1} of {num_epochs}, learning rate {lr:0.7f}")
    GetTime()

    # Scramble data
    idx = torch.randperm(X_train.shape[0])
    X_train = X_train[idx]
    y_train = y_train[idx]

    # # Shuffle in sampling dimension
    # idx = torch.randperm(num_chrom)
    # X_train[:,:-1] = X_train[:,:-1][:,:,idx]

    # Randomly muliply each example by 1 or -1
    # rand = torch.randint(0,2,size=(X_train.shape[0],1)) * 2 - 1
    # X_train *= rand

    for istart in range(0,X_train.shape[0],batch_size):

        iend = min(istart + batch_size, X_train.shape[0])

        # for k in range(istart, iend):  #get random index from y where y is nonzero
        #     b = torch.nonzero(y[k] - 1, as_tuple=False)
        #     random_index = tuple(b[torch.randint(0, b.shape[0],(1,)).item()].tolist())

        #     c = torch.zeros_like(y[k])
        #     c[random_index] = y[k,random_index] - 1
        #     c = c.unsqueeze(0)

        random_indices1 = torch.randint(0, input_size_processing, (iend - istart,))
        random_indices2 = torch.randint(0, n_embd_processing, (iend - istart,))

        y_batch = y_train[istart:iend]
        X_batch = X_train[istart:iend].to(device)

        output_info = torch.zeros_like(X_batch).long()

        for k in range(iend - istart):  # should be parallelizable
            valid_y = torch.nonzero(y_batch[k,:,random_indices2[k]] - 1)
            valid_index = valid_y[torch.randint(0, valid_y.shape[0], (1,)).item()]

            output_info[k,valid_index,random_indices2[k]] = y_batch[k,valid_index,random_indices2[k]] - 1

        y_batch = y_batch[torch.arange(iend - istart),random_indices1, random_indices2].to(device)
        random_indices1 = random_indices1.to(device)
        random_indices2 = random_indices2.to(device)

        optimizer.zero_grad()

        with training_context():
            try:
                y_pred = model(X_batch, random_indices1, random_indices2, output_info)
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                y_pred = model(X_batch, random_indices1, random_indices2, output_info)

            loss = criterion(y_pred, y_batch)
                             
        if mixed_precision:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        else:
            loss.backward()
            optimizer.step()

    lr *= lr_factor
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    loss = estimate_loss(min(num_estimate, X_test.shape[0]))

    if save_file.lower() != "none.pth" and loss < best_loss:
        best_loss = loss
        print("SAVING MODEL")
        torch.save(model.state_dict(), save_file)
