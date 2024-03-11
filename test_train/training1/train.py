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
model = SimpleNet()
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
    idx = torch.randperm(X.shape[0])
    idx = idx[:min(num_samples,len(idx))]
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

        for iter in range(ceil(num_samples / X.shape[0])): #it's possible to make this slightly more efficient
            for istart in range(iter * X.shape[0], min((iter + 1) * X.shape[0], num_samples), batch_size):
               
                iend = min(istart + batch_size, (iter + 1) * X.shape[0], num_samples)
                random_indices1 = torch.randint(0, input_size, (iend - istart,))
                ########
                random_indices1 = torch.ones((iend - istart,)).long() * (input_size_processing // 2)
                ########
                random_indices2 = torch.randint(0, n_embd_processing, (iend - istart,))
                random_indices3 = torch.randint(0, n_embd_processing, (iend - istart,))

                X_batch = X[istart % X.shape[0]:(iend - 1) % X.shape[0] + 1]
                y_batch = y[istart % X.shape[0]:(iend - 1) % X.shape[0] + 1]

            
                PGI = X_batch[torch.arange(iend - istart),:,random_indices2].to(device) #batch, input_size
                RGI = X_batch[torch.arange(iend - istart),:,random_indices3].to(device) #batch, input_size
                RGO = y_batch[torch.arange(iend - istart),:,random_indices3].to(device) #batch, input_size
                PGO = F.one_hot(random_indices1.long(), input_size_processing).to(device) #batch, input_size
                Pos = X_batch[torch.arange(iend - istart),:,-1].to(device) # batch, input_size
                

                y_pred[istart:iend] = model(PGI, PGO, RGI, RGO, Pos)
                y_true[istart:iend] = y_batch[torch.arange(iend - istart),random_indices1, random_indices2]

        if GPU_available:
            y_pred = y_pred.to("cuda")
        
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
        #######
        random_indices1 = torch.ones((iend - istart,)).long() * (input_size_processing // 2)
        #######
        random_indices2 = torch.randint(0, n_embd_processing, (iend - istart,))
        random_indices3 = torch.randint(0, n_embd_processing, (iend - istart,))

        y_batch = y_train[istart:iend]
        X_batch = X_train[istart:iend].to(device)
        
        PGI = X_batch[torch.arange(iend - istart),:,random_indices2].to(device) #batch, input_size
        RGI = X_batch[torch.arange(iend - istart),:,random_indices3].to(device) #batch, input_size
        RGO = y_batch[torch.arange(iend - istart),:,random_indices3].to(device) #batch, input_size
        PGO = F.one_hot(random_indices1.long(), input_size_processing).to(device) #batch, input_size
        Pos = X_batch[torch.arange(iend - istart),:,-1].to(device) # batch, input_size
        y_batch = y_batch[torch.arange(iend - istart),random_indices1, random_indices2].to(device) #batch

        optimizer.zero_grad()

        # for arr in [RGI.int(), RGO.int(), PGI.int(), y_train[istart:iend][torch.arange(iend - istart),:,random_indices2]]:
        #     print(arr[0])

        # print()

        with training_context():
            try:
                y_pred = model(PGI, PGO, RGI, RGO, Pos)
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                y_pred = model(PGI, PGO, RGI, RGO, Pos)

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

    loss = estimate_loss(num_estimate)

    if save_file.lower() != "none.pth" and loss < best_loss:
        best_loss = loss
        print("SAVING MODEL")
        torch.save(model.state_dict(), save_file)
