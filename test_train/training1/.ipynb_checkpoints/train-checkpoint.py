from processing import *
from models import *
import multiprocessing
import concurrent.futures
import torch
import torch.nn as nn
from ast import literal_eval
import sys

save_file = sys.argv[1] if sys.argv[1].endswith(".pth") else sys.argv[1] + ".pth"
data_dir = "../Data7"
GPU_available = torch.cuda.is_available()
print(GPU_available)
num_files = 50_000
num_epochs = 30
batch_size = 128
train_prop = 0.9
num_estimate = 5000
lr_start = 3e-3
lr_end = lr_start/100

GetMemory()
X = [convert_files(data_dir + "/sampled_genotypes/sample_" + str(i), data_dir + "/commands/command_" + str(i)) for i in range(num_files)] 

C = [convert_command_file1(data_dir + "/commands/command_" + str(i)) for i in range(num_files) if X[i] is not None]

X = [x for x in X if x is not None]


X = torch.tensor(X) - 1
X = X.reshape(-1, sample_width*num_chrom * 2)

print(X.shape)
C = torch.tensor(C)
C = C.reshape(-1, 13)

GetMemory()

y = torch.tensor([i%2 for i in range(1,X.shape[0] + 1)])

# Scramble data
torch.manual_seed(random_seed)
ind = int(train_prop * X.shape[0]) // 4
idx = torch.randperm(X.shape[0] // 4)
idx = [4*i     for i in idx[:ind]] + [4*i + 1 for i in idx[:ind]] \
    + [4*i + 2 for i in idx[:ind]] + [4*i + 3 for i in idx[:ind]] \
    + [4*i     for i in idx[ind:]] + [4*i + 1 for i in idx[ind:]] \
    + [4*i + 2 for i in idx[ind:]] + [4*i + 3 for i in idx[ind:]] 

X = X[idx]
y = y[idx]
C = C[idx]

# Split data
ind = int(train_prop * X.shape[0])
X_train, X_test = X[:ind], X[ind:]
y_train, y_test = y[:ind], y[ind:]
C_train, C_test = C[:ind], C[ind:]

GetMemory()
GetTime()

# lr variables
lr = lr_start
lr_factor = (lr_end/lr_start)**(1/(num_epochs - 1))

# Define network
model = BiasModel()
criterion = nn.BCELoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

# Define functions for getting random batch and calculating loss
def get_batch(split, num_samples):
    X, y, C = (X_train, y_train, C_train) if split == "train" else (X_test, y_test, C_test)
    idx = torch.randperm(X.shape[0])[:num_samples]
    X = X[idx]
    y = y[idx]
    C = C[idx]
    if GPU_available:
        return X.to("cuda"), y.to("cuda"), C.to("cuda")
    else:
        return X, y, C

@torch.no_grad()
def estimate_loss(num_samples):
    model.eval()
    for split in ["train", "test"]:
        X, y, C = get_batch(split, num_samples)
        y_pred = torch.zeros(num_samples)
        for i in range(0, num_samples, batch_size):
            try:
                X_batch = X[i:i+batch_size]
                y_pred[i:i+batch_size] = model(X_batch)
            except IndexError:
                X_batch = X[i:]
                y_pred[i:] = model(X_batch)

        if GPU_available:
            y_pred = y_pred.to("cuda")                       


        
        weak_epistasis = (C[:,7] < C[:,10])
        weak_count = weak_epistasis.sum().item()
        
        loss = criterion(y_pred, y.float())
        predictions = (y_pred >= 0.5).int()


        print()
        print(split, "full set:")

        num_correct = (predictions == y).sum().item()
        acc = num_correct/num_samples
        print(f"Loss {split}: {loss.item():0.5f}, Accuracy {split}: {acc:0.4f}")

        print("\t0\t1")
        for i in range(2):
            called_0 = torch.sum((predictions == 0) & (y == i)).item()
            called_1 = torch.sum((predictions == 1) & (y == i)).item()
            sums = called_0 + called_1
            called_0 /= sums
            called_1 /= sums
            print(i, f"{called_0:.3f}" , f"{called_1:.3f}", sep='\t')
        

            
        called_epistasic = (predictions == 1)
        
        print()
        print(split, "M and T evaluation")

        for i in range(8):
            minim = 0.1 + i/10
            maxum = 0.2 + i/10

            # admixture proportion
            subset_0 = ((C[:,1] >= minim) & (C[:,1] < maxum))

            for j in range(10):

                minim = 100 + j*50
                maxum = 150 + j*50

                # admixture time
                subset_1 = ((C[:,2] >= minim) & (C[:,2] < maxum))

                subset = (subset_0 & subset_1 & called_epistasic)

                #Counting proportion of those called epistasis which are correct

                subset_count = subset.sum().item()
                num_correct = ((predictions == y) & subset).sum().item()

                if subset_count == 0:
                    print("0",end='')
                    continue
                
                ratio = num_correct / subset_count

                if ratio < 0.5:
                    print(".",end='')
                elif ratio < 0.66:
                    print("-",end='')
                elif ratio < 0.80:
                    print("+",end='')
                elif ratio < 0.95:
                    print("$",end='')
                else:
                    print("#",end='')
            
            print()


        print()

        print(split, "epistrength 1 and 2 evaluation")

        for i in range(9):
            minim = 0.01 + i * 0.01
            maxum = 0.02 + i * 0.01

            subset_0 = ((C[:,7] >= minim) & (C[:,7] < maxum))

            for j in range(9):

                minim = 0.01 + j * 0.01
                maxum = 0.02 + j * 0.01

                subset_1 = ((C[:,11] >= minim) & (C[:,11] < maxum))

                subset = (subset_0 & subset_1 & called_epistasic)

                subset_count = subset.sum().item()
                num_correct = ((predictions == y) & subset).sum().item()

                if subset_count == 0:
                    print("0",end='')
                    continue
                
                ratio = num_correct / subset_count

                if ratio < 0.5:
                    print(".",end='')
                elif ratio < 0.66:
                    print("-",end='')
                elif ratio < 0.80:
                    print("+",end='')
                elif ratio < 0.95:
                    print("$",end='')
                else:
                    print("#",end='')
            
            print()


        print()

    model.train()

    return acc


if GPU_available:
    print("GPU is available.")
    model = model.to("cuda")
    criterion = criterion.to("cuda")
#    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:<enter-size-here>"
else:
    print("No GPU available. Running on CPU.")

GetMemory()
# estimate_loss(min(num_estimate, X_test.shape[0]))
#Training loop

#os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb: SIZE"

best_acc = 0
model.train()
for epoch in range(num_epochs):
    print("-----------------------------------------------------------------")
    print(f"Started training on epoch {epoch + 1} of {num_epochs}, learning rate {lr:0.7f}")
    GetTime()
    # Scramble data
    idx = torch.randperm(X_train.shape[0])
    X_train = X_train[idx]
    y_train = y_train[idx]

    # Shuffle in sampling dimension
    X_train = X_train.reshape(-1,2*sample_width,num_chrom)
    idx = torch.randperm(num_chrom)
    X_train = X_train[:,:,idx]

    # Randomly switch two sites 
    mask = torch.rand(X_train.shape[0]) < 0.5
    shift_indices = torch.fmod(torch.arange(2*sample_width) + sample_width, 2*sample_width)
    X_train[mask] = X_train[mask][:, shift_indices]


    X_train = X_train.reshape(-1,sample_width*num_chrom*2)
    # Randomly muliply each example by 1 or -1
    # rand = torch.randint(0,2,size=(X_train.shape[0],1)) * 2 - 1
    # X_train *= rand


    for ind in range(0,X_train.shape[0],batch_size):

        #print(torch.cuda.max_memory_allocated(),"and", torch.cuda.memory_allocated())

        try:
            X_batch = X_train[ind:ind+batch_size]
            y_batch = y_train[ind:ind+batch_size]
        except IndexError:
            X_batch = X_train[ind:]
            y_batch = y_train[ind:]

        if GPU_available:
            X_batch = X_batch.to("cuda")
            y_batch = y_batch.to("cuda")

        try:
            y_pred = model(X_batch)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            y_pred = model(X_batch)


        loss = criterion(y_pred, y_batch.float())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    lr *= lr_factor
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    acc = estimate_loss(min(num_estimate, X_test.shape[0]))
    # for child in model.children():
    #     if isinstance(child,nn.Linear):
    #         print("next")
    #         print("weight")
    #         print(child.weight)
    #         print("bias")
    #         print(child.bias)

    if save_file.lower() != "none.pth" and acc > best_acc:
        best_acc = acc
        print("SAVING MODEL")
        torch.save(model.state_dict(), save_file)
