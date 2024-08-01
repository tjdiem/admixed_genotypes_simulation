import torch
from globals import *
from math import e

population_size = 10000
num_generations = 32
admixture_proportion = 0.606

p_0a = 0.5
p_2a = 1 - p_0a

len_seq = 450
batch_size = 16

positions = torch.arange(950) / num_bp * 10000

positions_batch = positions.unsqueeze(0).expand(batch_size, -1)

lam = 2 * population_size * (1 - e ** (-num_generations / (2*population_size)))
lam_0 = admixture_proportion * lam
lam_1 = (1 - admixture_proportion) * lam
transition_ancestry_probs = torch.tensor([[p_0a, 0, p_2a],
                                [0, 1, 0],
                                [p_2a, 0, p_0a]]).to(device)


positions_diff = (positions - positions_batch[:, 475].unsqueeze(-1)).abs().to(device) # batch, len_seq + input_size
transition_00 = lam_1 / lam + (lam_0/lam) * torch.exp(-lam * positions_diff) # batch, len_seq + input_size - 1
transition_11 = lam_0 / lam + (lam_1/lam) * torch.exp(-lam * positions_diff)
transition_01 = 1 - transition_00
transition_10 = 1 - transition_11

transitions = torch.zeros((batch_size, len_seq + input_size - 1, num_classes, num_classes)).to(device)
transitions[:, :, 0, 0] = transition_00 ** 2
transitions[:, :, 0, 1] = transition_00 * transition_01 * 2
transitions[:, :, 0, 2] = transition_01 ** 2
transitions[:, :, 1, 0] = transition_00 * transition_10
transitions[:, :, 1, 1] = transition_00 * transition_11 + transition_01 * transition_10
transitions[:, :, 1, 2] = transition_11 * transition_01
transitions[:, :, 2, 0] = transition_10 ** 2
transitions[:, :, 2, 1] = transition_11 * transition_10 * 2
transitions[:, :, 2, 2] = transition_11 ** 2

print(transitions[0, 475])
print(transitions[0, 675])
print(transitions[0, 875])
# print(transitions[0, 875])

transitions = transitions @ transition_ancestry_probs

print(transitions[0, 475])
print(transitions[0, 675])
print(transitions[0, 875])