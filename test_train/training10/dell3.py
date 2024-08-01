import torch

num_bp = 50_000_000
len_seq = 16512
num_classes = 3
device = "cuda" if torch.cuda.is_available() else "cpu"
lam = 32
k1 = 27

positions = torch.load("temp_data/positions.pt").float().cpu() / num_bp
y = torch.load("temp_data/y.pt")
outs = torch.load("temp_data/y_pred.pt")

alpha = 1.5
outs = (outs + alpha) / (1 + num_classes * alpha)

positions_predicted = positions
positions_diff = positions_predicted[1:] - positions_predicted[:-1]

# below can be simplified by a lot if we are assuming admixture proportion of 0.5
transition_aa_haploid = 0.5 + 0.5 * torch.exp(-lam * positions_diff) # batch, len_seq + input_size - 1
transition_cc_haploid = 0.5 + 0.5 * torch.exp(-lam * positions_diff)
transition_ac_haploid = 1 - transition_aa_haploid
transition_ca_haploid = 1 - transition_cc_haploid

transitions = torch.zeros((len_seq - 1, num_classes, num_classes)).float().to(device)
transitions[:, 0, 0] = transition_aa_haploid ** 2
transitions[:, 0, 1] = transition_aa_haploid * transition_ac_haploid * 2
transitions[:, 0, 2] = transition_ac_haploid ** 2
transitions[:, 1, 0] = transition_aa_haploid * transition_ca_haploid
transitions[:, 1, 1] = transition_aa_haploid * transition_cc_haploid + transition_ac_haploid * transition_ca_haploid
transitions[:, 1, 2] = transition_cc_haploid * transition_ac_haploid
transitions[:, 2, 0] = transition_ca_haploid ** 2
transitions[:, 2, 1] = transition_cc_haploid * transition_ca_haploid * 2
transitions[:, 2, 2] = transition_cc_haploid ** 2

# relative prob of all predictions to the right matching
cum_probs_right = torch.zeros_like(outs)
cum_probs_right[-1] = outs[-1]
for i in range(outs.shape[0] - 2, -1, -1):
    P = outs[i] * (transitions[i] * cum_probs_right[i + 1].unsqueeze(0)).sum(dim=-1)
    cum_probs_right[i] = P / P.sum()
    # P *= (transitions[i] * outs[i].unsqueeze(0)).sum(dim=-1)
    # P /= P.sum()
    # cum_probs_right[i] = P

# relative prob of all predictions to the left matching
cum_probs_left = torch.zeros_like(outs)
cum_probs_left[0] = outs[0]
for i in range(1, outs.shape[0]):
    P = outs[i] * (transitions[i - 1] * cum_probs_left[i - 1].unsqueeze(0)).sum(dim=-1)
    cum_probs_left[i] = P / P.sum()

predictions = torch.zeros((len_seq, 3)).to(device)
for i in range(positions.shape[0]):
    position = positions[i]
    try:
        right_pos = torch.where(positions_predicted > position)[0][0]
        right_pred = cum_probs_right[right_pos]
        right_dist = (positions_predicted[right_pos] - position).item()
    except IndexError:
        right_pred = torch.tensor([1,1,1]).float().to(device)
        right_dist = 0.0

    try:
        left_pos = torch.where(positions_predicted < position)[0][-1]
        left_pred = cum_probs_left[left_pos]
        left_dist = (position - positions_predicted[left_pos]).item()
    except IndexError:
        left_pred = torch.tensor([1,1,1]).float().to(device)
        left_dist = 0.0

    try:
        middle_pos = torch.where(positions_predicted == position)[0][0]
        middle_pred = outs[middle_pos]
    except IndexError:
        middle_pred = torch.tensor([1,1,1]).float().to(device)

    
    positions_diff = torch.tensor([left_dist, right_dist]).to(device)
    transition_aa_haploid = 0.5 + 0.5 * torch.exp(-lam * positions_diff) # batch, len_seq + input_size - 1
    transition_cc_haploid = 0.5 + 0.5 * torch.exp(-lam * positions_diff)
    transition_ac_haploid = 1 - transition_aa_haploid
    transition_ca_haploid = 1 - transition_cc_haploid

    transitions = torch.zeros((2, num_classes, num_classes)).float().to(device)
    transitions[:, 0, 0] = transition_aa_haploid ** 2
    transitions[:, 0, 1] = transition_aa_haploid * transition_ac_haploid * 2
    transitions[:, 0, 2] = transition_ac_haploid ** 2
    transitions[:, 1, 0] = transition_aa_haploid * transition_ca_haploid
    transitions[:, 1, 1] = transition_aa_haploid * transition_cc_haploid + transition_ac_haploid * transition_ca_haploid
    transitions[:, 1, 2] = transition_cc_haploid * transition_ac_haploid
    transitions[:, 2, 0] = transition_ca_haploid ** 2
    transitions[:, 2, 1] = transition_cc_haploid * transition_ca_haploid * 2
    transitions[:, 2, 2] = transition_cc_haploid ** 2

    predictions[i] = (transitions[0] * left_pred.unsqueeze(0)).sum(dim=-1)
    predictions[i] *= (transitions[1] * right_pred.unsqueeze(0)).sum(dim=-1)
    predictions[i] *= middle_pred

    if i % 100 == 0:
        print(i)


# ap = 0.6060440728762055 
# for i in range(3):
#     print("unsmoothed: ", (outs.argmax(dim=-1) == i).sum().item() / len_seq)
#     print("smoothed: ", (outs2.argmax(dim=-1) == i).sum().item() / len_seq)
#     print("actual: ", (y == i).sum().item() / len_seq)
#     print("expected: ", end="")
#     if i == 0:
#         print(ap ** 2)
#     if i == 1:
#         print(2 * ap * (1 - ap))
#     if i == 2:
#         print((1 - ap) ** 2)

#     print()


y_pred_strict = predictions.argmax(dim=-1)
acc = (y_pred_strict == y).sum().item()/ y.shape[0]
print(f"Accuracy: {acc:0.6f}")


# idx_correct = (y_pred_strict == y).cpu()
# y_pred = outs2

# plt.hist(y_pred[idx_correct].amax(dim=-1).cpu())
# plt.hist(y_pred[~idx_correct].amax(dim=-1).cpu())

# fig, (ax1, ax2) = plt.subplots(1, 2)
# ax1.hist(y_pred[idx_correct].amax(dim=-1).cpu())
# ax2.hist(y_pred[~idx_correct].amax(dim=-1).cpu())
# plt.show()

# import matplotlib.pyplot as plt

# num_ind = 190000
# num_ind = min(num_ind, y_pred.shape[0])
# plt.scatter(torch.arange(num_ind), y_pred[:num_ind].amax(dim=-1).cpu())
# plt.scatter(torch.arange(y_pred.shape[0])[:num_ind][~idx_correct[:num_ind]], torch.zeros((num_ind - idx_correct[:num_ind].sum())))
# idx_sampled = torch.arange(16 * 20).long() * y_pred.shape[0] // (16 * 20)
# plt.scatter(idx_sampled, torch.full((16 * 20,), 0.25))
# plt.show()