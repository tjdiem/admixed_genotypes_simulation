import torch


a = torch.tensor([
[0,1,2,1],
[0,2,1,2], 
[0,1,1,1]
])

b = torch.nonzero(a - 1, as_tuple=False)
random_index = tuple(b[torch.randint(0, b.shape[0],(1,)).item()].tolist())

c = torch.zeros_like(a)
c[random_index] = a[random_index]

