import torch

num_bp = 50_000_000
len_seq = 16512
num_classes = 3
device = "cuda" if torch.cuda.is_available() else "cpu"
lam = 32
k1 = 27

positions = torch.load("temp_data/positions.pt").float().cpu() / num_bp
y = torch.load("temp_data/y.pt").cpu()
outs = torch.load("temp_data/y_pred.pt").cpu()

positions_diff = (positions - positions.unsqueeze(-1)).abs() # batch, len_seq + input_size

# below can be simplified by a lot if we are assuming admixture proportion of 0.5
transition_aa_haploid = 0.5 + 0.5 * torch.exp(-lam * positions_diff) # batch, len_seq + input_size - 1
transition_cc_haploid = 0.5 + 0.5 * torch.exp(-lam * positions_diff)
transition_ac_haploid = 1 - transition_aa_haploid
transition_ca_haploid = 1 - transition_cc_haploid

transitions = torch.zeros((len_seq, len_seq, num_classes, num_classes)).float()
transitions[:, :, 0, 0] = transition_aa_haploid ** 2
transitions[:, :, 0, 1] = transition_aa_haploid * transition_ac_haploid * 2
transitions[:, :, 0, 2] = transition_ac_haploid ** 2
transitions[:, :, 1, 0] = transition_aa_haploid * transition_ca_haploid
transitions[:, :, 1, 1] = transition_aa_haploid * transition_cc_haploid + transition_ac_haploid * transition_ca_haploid
transitions[:, :, 1, 2] = transition_cc_haploid * transition_ac_haploid
transitions[:, :, 2, 0] = transition_ca_haploid ** 2
transitions[:, :, 2, 1] = transition_cc_haploid * transition_ca_haploid * 2
transitions[:, :, 2, 2] = transition_cc_haploid ** 2

transitions[:, :, 0, 0] = 1
transitions[:, :, 0, 1] = 0
transitions[:, :, 0, 2] = 0
transitions[:, :, 1, 0] = 0
transitions[:, :, 1, 1] = 1
transitions[:, :, 1, 2] = 0
transitions[:, :, 2, 0] = 0
transitions[:, :, 2, 1] = 0
transitions[:, :, 2, 2] = 1

out_smoothed = (outs.unsqueeze(1).unsqueeze(1) @ transitions).squeeze(-2).float() #@ transition_ancestry_probs
position_weights = torch.exp(-k1 * lam * positions_diff).unsqueeze(-1) #hardcoded for now #increase factor as time goes on

outs2 = (out_smoothed * position_weights).sum(dim=0) / (position_weights.sum(dim=0))

ap = 0.6060440728762055 
for i in range(3):
    print("unsmoothed: ", (outs.argmax(dim=-1) == i).sum().item() / len_seq)
    print("smoothed: ", (outs2.argmax(dim=-1) == i).sum().item() / len_seq)
    print("actual: ", (y == i).sum().item() / len_seq)
    print("expected: ", end="")
    if i == 0:
        print(ap ** 2)
    if i == 1:
        print(2 * ap * (1 - ap))
    if i == 2:
        print((1 - ap) ** 2)

    print()


y_pred_strict = outs2.argmax(dim=-1)
acc = (y_pred_strict == y).sum().item()/ y.shape[0]
print(f"Accuracy: {acc:0.6f}")


idx_correct = (y_pred_strict == y).cpu()
y_pred = outs2

# plt.hist(y_pred[idx_correct].amax(dim=-1).cpu())
# plt.hist(y_pred[~idx_correct].amax(dim=-1).cpu())

# fig, (ax1, ax2) = plt.subplots(1, 2)
# ax1.hist(y_pred[idx_correct].amax(dim=-1).cpu())
# ax2.hist(y_pred[~idx_correct].amax(dim=-1).cpu())
# plt.show()

import matplotlib.pyplot as plt

num_ind = 190000
num_ind = min(num_ind, y_pred.shape[0])
plt.scatter(torch.arange(num_ind), y_pred[:num_ind].amax(dim=-1).cpu())
plt.scatter(torch.arange(y_pred.shape[0])[:num_ind][~idx_correct[:num_ind]], torch.zeros((num_ind - idx_correct[:num_ind].sum())))
idx_sampled = torch.arange(16 * 20).long() * y_pred.shape[0] // (16 * 20)
plt.scatter(idx_sampled, torch.full((16 * 20,), 0.25))
plt.show()