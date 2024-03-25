from globals import *
from processing import *
from models import *
import matplotlib.pyplot as plt

torch.manual_seed(random_seed)
idx = torch.randperm(num_files)
ind = int(train_prop * num_files)
idx = idx[ind:].tolist()


GetMemory()
Data = [convert_files(data_dir + "/sampled_genotypes/sample_" + str(i), data_dir + "/commands/command_" + str(i)) for i in idx] 

X = [xx for xx, _  in Data if xx is not None]
y = [yy for xx, yy in Data if xx is not None]

C = [convert_command_file1(data_dir + "/commands/command_" + str(i)) for j,i in enumerate(idx) if X[j] is not None]

X = torch.tensor(X) - 1 #(num usable files, num_chrom, num_samples)
X = X.transpose(-2, -1) #(num usable files, num_samples, num_chrom)
X = X[:, ::2,:]

y = torch.tensor(y) * 100


print(X.shape)
print(y.shape)

C = torch.tensor(C)
print(C.shape)

model = TransformerModel1()
model.load_state_dict(torch.load("model1.pth", map_location=device))
model = model.to(device)
model.eval()

criterion = nn.MSELoss()

with torch.no_grad():
    num_samples = X.shape[0]
    y_pred = torch.zeros(num_samples)
    for i in range(0, num_samples, batch_size):
        try:
            X_batch = X[i:i+batch_size].to(device)
            y_pred[i:i+batch_size] = model(X_batch)
        except IndexError:
            X_batch = X[i:].to(device)
            y_pred[i:] = model(X_batch)

    if GPU_available:
        y_pred = y_pred.to("cuda")                       

    y = y.to(device)
    loss = criterion(y_pred, y).item()
    avg_dist = (y_pred - y).abs().mean().item()

    print(f"Loss : {loss:0.5f}, Avg dist: {avg_dist:0.4f}")

    print()
    print("gene flow 1 and 2 evaluation")
    print("gene flow 2 varies horizontally")
    print()

    for j in range(10):

        minim = 0.001 * j
        maxum = 0.001 * j + 0.001

        subset = ((C[:,-1] >= minim) & (C[:,-1] < maxum))

        subset_count = subset.sum().item()
        avg_dist_subset = (y_pred[subset] - y[subset]).abs().mean().item()

        plt.scatter(y_pred)
        plt.show()
    
    print()
