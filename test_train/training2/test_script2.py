import torch

rows = 10
cols = 5
a = torch.arange(cols)
a = a.repeat(rows, 1)

for ind in range(a.shape[0]):
    a[ind] = a[ind, torch.randperm(cols)]

print(torch.randperm(cols))

# a = a.apply_(lambda row: row[torch.randperm(cols)])

print(a)