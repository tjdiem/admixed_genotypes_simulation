import torch
from math import e

v = torch.tensor([[0.8, 0.1, 0.1],
                  [0, 1, 0]])


d = torch.tensor([0,
                  0.003])


admixture_proportion = 0.5
population_size = 10_000
num_generations = 32
num_classes = 3

lam = 2 * population_size * (1 - e ** (-num_generations / (2*population_size)))
lam_a = admixture_proportion * lam
lam_c = (1 - admixture_proportion) * lam

transition_aa_haploid = lam_c / lam + (lam_a/lam) * torch.exp(-lam * d) # batch, len_seq + input_size - 1
transition_cc_haploid = lam_a / lam + (lam_c/lam) * torch.exp(-lam * d)
transition_ac_haploid = 1 - transition_aa_haploid
transition_ca_haploid = 1 - transition_cc_haploid

transitions = torch.zeros((len(d), num_classes, num_classes))
transitions[:, 0, 0] = transition_aa_haploid ** 2
transitions[:, 0, 1] = transition_aa_haploid * transition_ac_haploid * 2
transitions[:, 0, 2] = transition_ac_haploid ** 2
transitions[:, 1, 0] = transition_aa_haploid * transition_ca_haploid
transitions[:, 1, 1] = transition_aa_haploid * transition_cc_haploid + transition_ac_haploid * transition_ca_haploid
transitions[:, 1, 2] = transition_cc_haploid * transition_ac_haploid
transitions[:, 2, 0] = transition_ca_haploid ** 2
transitions[:, 2, 1] = transition_cc_haploid * transition_ca_haploid * 2
transitions[:, 2, 2] = transition_cc_haploid ** 2

P = (transitions * v.unsqueeze(-2)).sum(dim=-1)
P = P.prod(dim=0)
P /= P.sum()

print(P)



