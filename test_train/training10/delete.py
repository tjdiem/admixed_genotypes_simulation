import torch
from math import e

# # y = y[::2][:49,:450]
# y = y[::2][:49]
# print(y.shape)
# num_individuals = 49
# num_tracts = (y[:,:- 1] != y[:, 1:]).sum().item() + num_individuals
# print(num_tracts)
# avg_len_transition = num_individuals * (5.8413e-02 - 5.1200e-06) / num_tracts
# print(avg_len_transition)
# num_generations = 1 / (2 * avg_len_transition)
# print(num_generations)
# exit()

# Constant for all simulations
num_bp = 50_000_000
recombination_rate = 1 / num_bp # this assumes recombination rate is constant along chromosome, for chromosome of 50M base pairs
population_size = 10_000 # from looking at demography file. Should we vary this parameter?

# Specific to simulation folder 11, id 5
num_generations = 32
admixture_proportion = 0.6060440728762055

lam = 2 * population_size * (1 - e ** (-num_generations / (2*population_size)))

lam_0 = admixture_proportion * lam
lam_1 = (1 - admixture_proportion) * lam

P = torch.tensor([[1 - lam_0 * recombination_rate, lam_0 * recombination_rate],
                  [lam_1 * recombination_rate, 1 - lam_1 * recombination_rate]], 
                  dtype=torch.float64)


print(P[0,0].item())
print(P[0,1].item())
print(P[0,0].item() + P[0,1].item())


l = int(0.05 * num_bp)
print(torch.matrix_power(P, l))

print(lam_1 / lam + lam_0 * e ** (-recombination_rate * l * lam)/ lam) 