from globals import *
from processing import *
from training_functions import *
from kmodels2 import KNet4

model_name = "model_0.pth"
num_probs_test = [0]
num_estimate = 1000
splits_test = ["train", "test"]
num_files = 270

# Load Data
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

# Define network
model = KNet4()
model.load_state_dict(torch.load(model_name, map_location=torch.device(device)))
model = model.to(device)
criterion = nn.KLDivLoss(reduction='batchmean')

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


model.eval()
with torch.no_grad():
    for split in splits_test:
        for num_probs in num_probs_test: # 0, 2,4 #hardcoded for now
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

