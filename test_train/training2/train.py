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

X = torch.tensor(X).float() - 1 #(num usable files, len_chrom, num_samples)
y = torch.tensor(y)

print(X.shape)
print(y.shape)

# for i in range(-1, 2):
#     print((X == i).sum().item())  # this gives weird results 


GetMemory()

# Scramble data
torch.manual_seed(random_seed)
idx = torch.randperm(X.shape[0])
X = X[idx]
y = y[idx]

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
model = ConvNet2()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

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

                X_batch = X[istart % X.shape[0]:(iend - 1) % X.shape[0] + 1].to(device)
                y_batch = y[istart % X.shape[0]:(iend - 1) % X.shape[0] + 1]

                I_batch = torch.arange(y_batch.shape[2]).long()
                I_batch = I_batch.repeat(y_batch.shape[0], 1)

                for i in range(I_batch.shape[0]):  #parallelize this
                    I_batch[i] = I_batch[i, torch.randperm(y_batch.shape[2])]

                for i in range(I_batch.shape[0]):  #parallelize this
                    X_batch[i,:,:-1] = X_batch[i,:,I_batch[i]]
                    y_batch[i] = y_batch[i,:,I_batch[i]]

                y_batch_out = y_batch[:,y_batch.shape[1] // 2, 0].to(device)
                y_batch_in = y_batch.float().to(device) - 1

                # IOI = (y_batch_out != 1) ###
                # y_batch_in = y_batch_in[IOI]  ###
                # y_batch_out = y_batch_out[IOI] ###
                # X_batch = X_batch[IOI]   ###

                # if X_batch.shape[0] == 0:  ###
                #     continue   ###

                # y_batch_in[:,0:,0] = 0

                y_pred[istart:iend] = model(X_batch, y_batch_in)
                y_true[istart:iend] = y_batch_out

                # y_pred[istart:istart+X_batch.shape[0]] = model(X_batch, y_batch_in) ###
                # y_true[istart:istart+X_batch.shape[0]] = y_batch_out ###

        # valid_indices = (y_pred.sum(dim=-1) != 0)  ###
        # y_pred = y_pred[valid_indices]   ###
        # y_true = y_true[valid_indices]   ###

        if GPU_available:
            y_pred = y_pred.to("cuda")
        
        loss = criterion(y_pred, y_true).item()
        predictions = y_pred.argmax(dim=-1)
        accuracy = (predictions == y_true).sum().item() / y_true.shape[0]

        print(y_pred)
        print(y_true)
        print(predictions)

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

GetMemory()

# estimate_loss(num_estimate)
# exit()

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

        y_batch = y_train[istart:iend]
        X_batch = X_train[istart:iend].to(device)

        I_batch = torch.arange(y_batch.shape[2]).long()
        I_batch = I_batch.repeat(y_batch.shape[0], 1)

        for i in range(I_batch.shape[0]):  #parallelize this
            I_batch[i] = I_batch[i, torch.randperm(y_batch.shape[2])]

        for i in range(I_batch.shape[0]):  #parallelize this
            X_batch[i,:,:-1] = X_batch[i,:,I_batch[i]]
            y_batch[i] = y_batch[i,:,I_batch[i]]

        y_batch_out = y_batch[:,y_batch.shape[1] // 2, 0].to(device)
        y_batch_in = y_batch.float().to(device) - 1


        # IOI = (y_batch_out != 1) ###
        # y_batch_in = y_batch_in[IOI] ### 
        # y_batch_out = y_batch_out[IOI] ###
        # X_batch = X_batch[IOI] ###

        # if X_batch.shape[0] == 0:  ###
        #     continue  ###
        # y_batch_in[:,0:,0] = 0

        optimizer.zero_grad()

        with training_context():

            # print(y_batch_in[:,y_batch.shape[1] // 2, 0] + 1 == y_batch_out)
            if checkpointing:
                X_batch.requires_grad_(True)
                y_batch_in.requires_grad_(True)
                try:
                    y_pred = checkpoint(model.forward, X_batch, y_batch_in)
                except torch.cuda.OutOfMemoryError:
                    torch.cuda.empty_cache()
                    y_pred = checkpoint(model.forward, X_batch, y_batch_in)

            else:
                try:
                    y_pred = model(X_batch, y_batch_in)
                except torch.cuda.OutOfMemoryError:
                    torch.cuda.empty_cache()
                    y_pred = model(X_batch, y_batch_in)

            # print(y_pred)
            # print(y_batch_out)
            loss = criterion(y_pred, y_batch_out)

            # print(loss)
            # print()

        if mixed_precision:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        else:
            loss.backward()
            optimizer.step()

    lr *= lr_factor
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    if (epoch + 1) % eval_interval == 0:

        loss = estimate_loss(num_estimate)

        if save_file.lower() != "none.pth" and loss < best_loss:
            best_loss = loss
            print("SAVING MODEL")
            torch.save(model.state_dict(), save_file)
