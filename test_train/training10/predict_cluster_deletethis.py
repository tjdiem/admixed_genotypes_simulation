import torch
from processing import *

random_file = 4

len_seq = 750

X, positions = convert_panel(panel_dir + "panel_" + str(random_file))
X = torch.tensor(X).to(device)[:49, :len_seq] # n_ind_adm, input_size
positions = torch.tensor(positions).to(device)[:len_seq]

y = convert_split(split_dir + "split_" + str(random_file), positions)
y = torch.tensor(y) # 2 * n_ind_adm, input_size
y = (y[::2] + y[1::2])[:49, :len_seq].to(device) # unphase ancestry labels # same shape as X

print(F.one_hot(y).sum(dim=0).sum(dim=0))
exit()

# print(y[13]) # 1 - good
# print(y[26]) # 1 - good
# print(y[31]) # 2 - bad
# print(y[46]) # 2 - good
# exit()

print(y.shape)

y_pred0 = torch.load("tempdir/predictions_chunked0.pt")[:, input_size // 2: -(input_size // 2)]
y_pred1 = torch.load("tempdir/predictions_chunked1.pt")[:, input_size // 2: -(input_size // 2)]

print(y_pred0[0, 250:])
print(y_pred1[0, :250])

y_pred0_strict = torch.argmax(y_pred0, dim=-1)
y_pred1_strict = torch.argmax(y_pred1, dim=-1)


print((y_pred0_strict == y[:, :500]).float().mean())
print((y_pred0_strict == 2 - y[:, :500]).float().mean())
print((y_pred1_strict == y[:, 250:]).float().mean())
print((y_pred1_strict == 2 - y[:, 250:]).float().mean())
