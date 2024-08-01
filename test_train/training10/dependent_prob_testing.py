import torch
from math import e

v = torch.tensor([[0.8, 0.1, 0.1],
                  [0.6, 0.2, 0.2],
                  [0.6, 0.3, 0.1],
                  [0.5, 0.3, 0.2]])


d = torch.tensor([0.005,
                  0.01,
                  0.02,
                  0.04])


admixture_proportion = 0.5
population_size = 10_000
num_generations = 32
num_classes = 3

lam = 2 * population_size * (1 - e ** (-num_generations / (2*population_size)))
lam_a = admixture_proportion * lam
lam_c = (1 - admixture_proportion) * lam

# transition_aa_haploid = lam_c / lam + (lam_a/lam) * torch.exp(-lam * d) # batch, len_seq + input_size - 1
# transition_cc_haploid = lam_a / lam + (lam_c/lam) * torch.exp(-lam * d)
# transition_ac_haploid = 1 - transition_aa_haploid
# transition_ca_haploid = 1 - transition_cc_haploid

# transitions = torch.zeros((len(d), num_classes, num_classes))
# transitions[:, 0, 0] = transition_aa_haploid ** 2
# transitions[:, 0, 1] = transition_aa_haploid * transition_ac_haploid * 2
# transitions[:, 0, 2] = transition_ac_haploid ** 2
# transitions[:, 1, 0] = transition_aa_haploid * transition_ca_haploid
# transitions[:, 1, 1] = transition_aa_haploid * transition_cc_haploid + transition_ac_haploid * transition_ca_haploid
# transitions[:, 1, 2] = transition_cc_haploid * transition_ac_haploid
# transitions[:, 2, 0] = transition_ca_haploid ** 2
# transitions[:, 2, 1] = transition_cc_haploid * transition_ca_haploid * 2
# transitions[:, 2, 2] = transition_cc_haploid ** 2

# P = (transitions * v.unsqueeze(-2)).sum(dim=-1)
# P = P.prod(dim=0)
# P /= P.sum()

# print(P)

positions_diff = torch.cat(((d[0] - 0).unsqueeze(0), d[1:] - d[:-1]))
print(positions_diff)

transition_aa_haploid = lam_c / lam + (lam_a/lam) * torch.exp(-lam * positions_diff) # batch, len_seq + input_size - 1
transition_cc_haploid = lam_a / lam + (lam_c/lam) * torch.exp(-lam * positions_diff)
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

P = torch.tensor([1,1,1]).float()
cum_probs = torch.zeros((len(d), 3))
for i in range(len(d) - 1, -1, -1):
    P *= (transitions[i] * v[i].unsqueeze(0)).sum(dim=-1)
    print(transitions[i])
    print(v[i])
    print()
    P /= P.sum()
    cum_probs[i] = P

print(cum_probs)
    # P *= (transitions[-2] * v[-1].unsqueeze(0)).sum(dim=-1)
    # print(P)



# v1hat = v1
# v2hat = v2 @ transitions
# v2hat_given_v1hat = v2hat / (v1hat @ transitions @ transitions)
# v2hat_given_v1hat = v2hat / v2hat.sum()

# P = v1hat * v2hat_given_v1hat
# P /= P.sum()

# print(v1hat)
# print(v2hat_given_v1hat)
# print(P)

