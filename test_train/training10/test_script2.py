import torch

mask = torch.tensor([0, 0, 1, 0, 1, 0, 1]).bool()
A = torch.tensor([10, 20, 30, 40, 50, 60, 70])
P = torch.tensor([0.01, 0.03, 0.04, 0.05, 0.08, 0.11, 0.12])

# result = torch.cat((A[mask], torch.tensor([0])))
# mask = torch.cumsum(mask, 0)

# result = result[mask]

# print(result)

A_filtered = torch.cat((A[mask].flip(0), torch.tensor([0])))

mask = torch.cumsum(mask.flip(0), 0).flip(0)
result = A - A_filtered[mask]

print(result)

# P_mask = torch.cat((P[mask].flip(0), torch.tensor([0])))

# mask = torch.cumsum(mask.flip(0), 0).flip(0)
# result = P - P_mask[mask]
# print(P_mask)
