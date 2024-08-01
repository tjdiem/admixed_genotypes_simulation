import torch
out1 = torch.tensor([0.7, 0.2, 0.1])
out2 = torch.tensor([0.7, 0.2, 0.1])
positions = torch.tensor([0, 0.005])
lam = 32


positions_predicted = positions
positions_diff = positions_predicted[1:] - positions_predicted[:-1]

len_seq = len(positions)
num_classes = 3
# below can be simplified by a lot if we are assuming admixture proportion of 0.5
transition_aa_haploid = 0.5 + 0.5 * torch.exp(-lam * positions_diff) # batch, len_seq + input_size - 1
transition_cc_haploid = 0.5 + 0.5 * torch.exp(-lam * positions_diff)
transition_ac_haploid = 1 - transition_aa_haploid
transition_ca_haploid = 1 - transition_cc_haploid

transitions = torch.zeros((len_seq - 1, num_classes, num_classes)).float()
transitions[:, 0, 0] = transition_aa_haploid ** 2
transitions[:, 0, 1] = transition_aa_haploid * transition_ac_haploid * 2
transitions[:, 0, 2] = transition_ac_haploid ** 2
transitions[:, 1, 0] = transition_aa_haploid * transition_ca_haploid
transitions[:, 1, 1] = transition_aa_haploid * transition_cc_haploid + transition_ac_haploid * transition_ca_haploid
transitions[:, 1, 2] = transition_cc_haploid * transition_ac_haploid
transitions[:, 2, 0] = transition_ca_haploid ** 2
transitions[:, 2, 1] = transition_cc_haploid * transition_ca_haploid * 2
transitions[:, 2, 2] = transition_cc_haploid ** 2

print(transitions)
