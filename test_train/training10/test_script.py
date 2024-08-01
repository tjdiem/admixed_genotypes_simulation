import torch
import torch.nn.functional as F

OHE_values = torch.arange(3**4).reshape(3, 3, 3, 3) # SOI value, ref value, class we are in, labeled class of ref
OHE_values[:, :, 2, :] = torch.flip(OHE_values[:, :, 0, :], dims=(2,))
unique_elements, inverse_indices = torch.unique(OHE_values, return_inverse=True)
OHE_values = torch.arange(len(unique_elements))[inverse_indices]
OHE_values = OHE_values.flatten()

n_embd = 54
assert n_embd == OHE_values.max() + 1

# SOI = torch.randint(0, 3, (4, 501))
# refs = torch.randint(0, 3, (4, 48, 501))
# labels = torch.randint(0, 3, (4, 48, 501))

# labels = F.one_hot(labels, num_classes=3)
# refs = F.one_hot(refs, num_classes=3)

# labels = labels.unsqueeze(-1).transpose(-2, -1)
# refs = refs.unsqueeze(-1)


a = torch.tensor([0,0,1])
b = torch.tensor([0,1,0])
c = torch.tensor([0,1,0])

a = a.unsqueeze(-1).unsqueeze(-1).repeat(1, 3, 3).flatten()
b = b.unsqueeze(0).unsqueeze(-1).repeat(3, 1, 3).flatten()
c = c.unsqueeze(0).unsqueeze(0).repeat(3, 3, 1).flatten()

# c + b * 3 + a * 9
out = a * b * c
print(out.argmax())
print(out)