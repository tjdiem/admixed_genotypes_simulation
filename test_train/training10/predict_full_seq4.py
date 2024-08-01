from globals import *
from processing import *
from kmodels2 import KNet4
from math import e, ceil
import matplotlib.pyplot as plt

@torch.no_grad()
def predict_full_sequence(model, SOI, positions, refs, labels, max_batch_size=batch_size, num_generations=None, population_size=None):
    # SOI     #len_seq
    # pos     #len_seq
    # refs    #n_refs, len_seq
    # labels  #n_refs, len_seq

    admixture_proportion = 0.5

    if population_size is None:
        population_size = 10_000

    # this approaches num_generations as population_size approaches inf
    lam = 2 * population_size * (1 - e ** (-num_generations / (2*population_size)))
    lam_a = admixture_proportion * lam
    lam_c = (1 - admixture_proportion) * lam

    positions = positions / num_bp

    start_pos = 0
    end_pos = 1 ####### hardcoded for now!!

    assert SOI.shape == positions.shape
    assert SOI.shape[0] == refs.shape[1]
    assert refs.shape == labels.shape[:2]

    n_refs, len_seq = refs.shape

    padding = torch.full((input_size // 2,), -1).to(device)
    SOI_padded = torch.cat((padding, SOI, padding), dim=0)
    positions_padded = torch.cat((padding, positions, padding), dim=0)

    padding = torch.full((n_refs, input_size // 2), -1).to(device)
    refs_padded = torch.cat((padding, refs, padding), dim=-1)

    padding = torch.zeros((n_refs, input_size // 2, num_classes)).to(device)
    labels_padded = torch.cat((padding, labels, padding), dim=1)

    predictions = torch.zeros((len_seq, num_classes)).to(device)

    predicted_idx = torch.zeros((len_seq)).bool()

    positions_next = torch.arange(batch_size).to(device) * ((end_pos - start_pos)/ batch_size) + start_pos + 1/(2*batch_size)
    positions_diff = (positions - positions_next.unsqueeze(-1)).abs() # batch, len_seq
    # start - ((pos - start) * 1/3 )
    # 4/3 * start - 1/3 * pos

    # end + ((end - pos) * 1/3)
    # 4/3 * end - 1/3 * pos


    num_iterations = 20
    temperature = 0.8
    outputs = torch.zeros((len_seq, 3)).float().to(device)
    idx_sampled = torch.arange(batch_size * num_iterations).long() * len_seq // (batch_size * num_iterations)
    idx_sampled = torch.randperm(len_seq)[:num_iterations * batch_size]#.sort()[0]
    for iteration in range(num_iterations):

        # idx_next = (torch.arange(batch_size) * (end - start) / batch_size + start).long()
        idx_next = idx_sampled[iteration * batch_size: (iteration + 1) * batch_size] 
        predicted_idx[idx_next] = 1
    
        # faster way to index this?
        SOI_batch = torch.stack([SOI_padded[idx_next[i]:idx_next[i]+input_size] for i in range(batch_size)])
        positions_batch = torch.stack([positions_padded[idx_next[i]:idx_next[i]+input_size] for i in range(batch_size)])
        refs_batch = torch.stack([refs_padded[:, idx_next[i]:idx_next[i]+input_size] for i in range(batch_size)])
        labels_batch = torch.stack([labels_padded[:, idx_next[i]:idx_next[i]+input_size] for i in range(batch_size)])
        params_batch = torch.full((batch_size, 6), num_generations).to(device)  # this is messy
        
        output = model(SOI_batch, refs_batch, labels_batch, positions_batch, params_batch)
        output = F.softmax(output, dim=-1) # batch, num_classes

        outputs[idx_next] = output

    outputs_predicted = outputs[predicted_idx]
    outputs_predicted = F.softmax(outputs_predicted.log() / temperature, dim=-1)
    # outputs_predicted = (outputs_predicted + alpha) / (1 + num_classes * alpha)

    positions_predicted = positions[predicted_idx] # (num_predictions)
    positions_diff = positions_predicted[1:] - positions_predicted[:-1]

    # below can be simplified by a lot if we are assuming admixture proportion of 0.5
    transition_aa_haploid = lam_c / lam + (lam_a/lam) * torch.exp(-lam * positions_diff) # batch, len_seq + input_size - 1
    transition_cc_haploid = lam_a / lam + (lam_c/lam) * torch.exp(-lam * positions_diff)
    transition_ac_haploid = 1 - transition_aa_haploid
    transition_ca_haploid = 1 - transition_cc_haploid
    
    transitions = torch.zeros((batch_size * num_iterations - 1, num_classes, num_classes)).float().to(device)
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
    cum_probs_right = torch.zeros_like(outputs_predicted)
    cum_probs_right[-1] = outputs_predicted[-1]
    for i in range(outputs_predicted.shape[0] - 2, -1, -1):
        P = outputs_predicted[i] * (transitions[i] * cum_probs_right[i + 1].unsqueeze(0)).sum(dim=-1)
        cum_probs_right[i] = P / P.sum()

    # relative prob of all predictions to the left matching
    cum_probs_left = torch.zeros_like(outputs_predicted)
    cum_probs_left[0] = outputs_predicted[0]
    for i in range(1, outputs_predicted.shape[0]):
        P = outputs_predicted[i] * (transitions[i - 1] * cum_probs_left[i - 1].unsqueeze(0)).sum(dim=-1)
        cum_probs_left[i] = P / P.sum()

    idx_expanded = torch.cumsum(predicted_idx, 0)
    cum_probs_right = torch.cat((cum_probs_right, torch.ones((1, num_classes)).to(device)))  ### !
    right_pred1 = cum_probs_right[idx_expanded]

    positions_predicted_padded = torch.cat((positions_predicted, torch.full((1,), -1).to(device)))
    right_dist1 = positions_predicted_padded[idx_expanded] - positions
    right_dist1[right_dist1 < 0] = 0 ##!!

    idx_expanded = torch.cumsum(predicted_idx.flip(0), 0).flip(0)# - predicted_idx.int().flip(0)
    cum_probs_left_copy = cum_probs_left.clone().flip(0)
    cum_probs_left_copy = torch.cat((cum_probs_left_copy, torch.ones((1, num_classes)).to(device)))
    left_pred1 = cum_probs_left_copy[idx_expanded]

    positions_predicted_padded = torch.cat((positions_predicted.flip(0), torch.full((1,), -1).to(device)))
    left_dist1 = positions - positions_predicted_padded[idx_expanded]
    left_dist1[left_dist1 > 1] = 0 ##!!

    middle_pred1 = torch.ones((len_seq, num_classes)).float().to(device)
    middle_pred1[predicted_idx] = outputs_predicted

    transition_aa_haploid = lam_c / lam + (lam_a/lam) * torch.exp(-lam * right_dist1) # batch, len_seq + input_size - 1
    transition_cc_haploid = lam_a / lam + (lam_c/lam) * torch.exp(-lam * right_dist1)
    transition_ac_haploid = 1 - transition_aa_haploid
    transition_ca_haploid = 1 - transition_cc_haploid
    
    transitions_right = torch.zeros((len_seq, num_classes, num_classes)).float().to(device)
    transitions_right[:, 0, 0] = transition_aa_haploid ** 2
    transitions_right[:, 0, 1] = transition_aa_haploid * transition_ac_haploid * 2
    transitions_right[:, 0, 2] = transition_ac_haploid ** 2
    transitions_right[:, 1, 0] = transition_aa_haploid * transition_ca_haploid
    transitions_right[:, 1, 1] = transition_aa_haploid * transition_cc_haploid + transition_ac_haploid * transition_ca_haploid
    transitions_right[:, 1, 2] = transition_cc_haploid * transition_ac_haploid
    transitions_right[:, 2, 0] = transition_ca_haploid ** 2
    transitions_right[:, 2, 1] = transition_cc_haploid * transition_ca_haploid * 2
    transitions_right[:, 2, 2] = transition_cc_haploid ** 2


    transition_aa_haploid = lam_c / lam + (lam_a/lam) * torch.exp(-lam * left_dist1) # batch, len_seq + input_size - 1
    transition_cc_haploid = lam_a / lam + (lam_c/lam) * torch.exp(-lam * left_dist1)
    transition_ac_haploid = 1 - transition_aa_haploid
    transition_ca_haploid = 1 - transition_cc_haploid
    
    transitions_left = torch.zeros((len_seq, num_classes, num_classes)).float().to(device)
    transitions_left[:, 0, 0] = transition_aa_haploid ** 2
    transitions_left[:, 0, 1] = transition_aa_haploid * transition_ac_haploid * 2
    transitions_left[:, 0, 2] = transition_ac_haploid ** 2
    transitions_left[:, 1, 0] = transition_aa_haploid * transition_ca_haploid
    transitions_left[:, 1, 1] = transition_aa_haploid * transition_cc_haploid + transition_ac_haploid * transition_ca_haploid
    transitions_left[:, 1, 2] = transition_cc_haploid * transition_ac_haploid
    transitions_left[:, 2, 0] = transition_ca_haploid ** 2
    transitions_left[:, 2, 1] = transition_cc_haploid * transition_ca_haploid * 2
    transitions_left[:, 2, 2] = transition_cc_haploid ** 2  

    predictions = (transitions_left * left_pred1.unsqueeze(1)).sum(dim=-1) * (transitions_right * right_pred1.unsqueeze(1)).sum(dim=-1) * middle_pred1

    return predictions / predictions.sum(dim=-1, keepdim=True)


model = eval(model_name)()
model = model.to(device)
model.load_state_dict(torch.load("full_seq_refalt.pth", map_location=torch.device(device)))
model.eval()

random_file = 0 if human_data else 5

torch.manual_seed(409)
random.seed(409)

with open(parameters_dir + "parameter_" + str(random_file)) as f:
    admixture_proportion, num_generations, *_ = f.readlines()
    admixture_proportion = float(admixture_proportion)
    num_generations = int(num_generations)

X, positions = convert_panel(panel_dir + "panel_" + str(random_file))
X = torch.tensor(X).to(device).squeeze(0) # len_seq
positions = torch.tensor(positions).to(device) # len_seq

print(X.shape)
print(positions.shape)

y = convert_split(split_dir + "split_" + str(random_file), positions)
y = torch.tensor(y).to(device) # 2 * n_ind_adm, len_seq
y = y[0] + y[1] # len_seq

print(y.shape)

refA, refB, _ = convert_panel_template(panel_template_dir + "panel_template_" + str(random_file)) 

refA = torch.tensor(refA).to(device) #num_files, n_ind_pan, input_size
refB = torch.tensor(refB).to(device) #num_files, n_ind_pan, input_size
refs = torch.zeros((num_classes, n_ind_pan_model, refA.shape[-1])).to(device)
refs[0] = refA[:2 * n_ind_pan // 6 * 2:2] + refA[1:2 * n_ind_pan // 6 * 2:2]
refs[2] = refB[:2 * n_ind_pan // 6 * 2:2] + refB[1:2 * n_ind_pan // 6 * 2:2]
refs[1] = refA[-(n_ind_pan // 6 * 2):] + refB[-(n_ind_pan // 6 * 2):]
refs[1] = -1 ####################!!!!!!!!!!!!!!!!!!!!
refs = refs.reshape(n_ind_max, -1)

labels = torch.arange(num_classes).to(device).unsqueeze(-1).unsqueeze(-1)
labels = labels.repeat(1, n_ind_pan_model, refs.shape[-1])

labels = F.one_hot(labels.long(), num_classes=num_classes)
labels[1] = 0 ######################!!!!!!!!!!!!!!!!!!!
labels = labels.reshape(n_ind_max, -1, num_classes)

print(refs.shape)
print(labels.shape)


print()
t1 = time.time()
y_pred = predict_full_sequence(model, X, positions, refs, labels, num_generations=num_generations)
time_spent = time.time() - t1
print(f"Time: {time_spent:0.3f}")

y_pred_strict = y_pred.argmax(dim=-1)
acc = (y_pred_strict == y).sum().item()/ y.shape[0]
print(f"Accuracy: {acc:0.6f}")

idx_correct = (y_pred_strict == y).cpu()

plt.hist(y_pred[idx_correct].amax(dim=-1).cpu())
plt.hist(y_pred[~idx_correct].amax(dim=-1).cpu())

fig, (ax1, ax2) = plt.subplots(1, 2)
ax1.hist(y_pred[idx_correct].amax(dim=-1).cpu())
ax2.hist(y_pred[~idx_correct].amax(dim=-1).cpu())
plt.show()

num_ind = 190000
num_ind = min(num_ind, y_pred.shape[0])
plt.scatter(torch.arange(num_ind), y_pred[:num_ind].amax(dim=-1).cpu())
plt.scatter(torch.arange(y_pred.shape[0])[:num_ind][~idx_correct[:num_ind]], torch.zeros((num_ind - idx_correct[:num_ind].sum())))
idx_sampled = torch.arange(batch_size * 20).long() * y_pred.shape[0] // (batch_size * 20)
plt.scatter(idx_sampled, torch.full((batch_size * 20,), 0.25))
plt.show()


"""
run full predictions inference every so often instead of at the end
stopping criteria: when the change in predictions is small (could be probabilities or argmax)
find ideal temperature based on sampling density
find best indexes to sample in next iteration
"""


if False:
    for i in range(400):
        print(predicted_idx[i].int().item(), end="-")
        print(idx_expanded[i].item(), end = "  ")

    print(predicted_idx[3299:3306])
    print(idx_expanded[3299:3306])
    print(predicted_idx[6602:6609])
    print(idx_expanded[6602:6609])

    # result = torch.cat((A[mask], torch.tensor([0])))
    # mask = torch.cumsum(mask, 0)

    # result = result[mask]

    for i in range(positions.shape[0]):
        position = positions[i]
        try:
            right_pos = torch.where(positions_predicted > position)[0][0]
            right_pred = cum_probs_right[right_pos]
            right_dist = (positions_predicted[right_pos] - position).item()
        except IndexError:
            right_pred = torch.tensor([1,1,1]).float().to(device)
            right_dist = 0.0

        if not torch.equal(right_pred, right_pred1[i]):
            print("right")
            print(i)
            print()

        if right_dist != right_dist1[i].item():
            print("right dist")
            print(i)
            print()

        try:
            left_pos = torch.where(positions_predicted < position)[0][-1]
            left_pred = cum_probs_left[left_pos]
            left_dist = (position - positions_predicted[left_pos]).item()
        except IndexError:
            left_pred = torch.tensor([1,1,1]).float().to(device)
            left_dist = 0.0


        if not torch.equal(left_pred, left_pred1[i]):
            print("left")
            print(i)
            print(left_pred, left_pred1[i])
            print()

        if left_dist != left_dist1[i].item():
            print("left dist")
            print(i)
            print()

        try:
            middle_pos = torch.where(positions_predicted == position)[0][0]
            middle_pred = outputs_predicted[middle_pos]
        except IndexError:
            middle_pred = torch.tensor([1,1,1]).float().to(device)

        if not torch.equal(middle_pred, middle_pred1[i]):
            print("middle")
            print(i)
            print()
        
        positions_diff = torch.tensor([left_dist, right_dist]).to(device)
        transition_aa_haploid = lam_c / lam + (lam_a/lam) * torch.exp(-lam * positions_diff) # batch, len_seq + input_size - 1
        transition_cc_haploid = lam_a / lam + (lam_c/lam) * torch.exp(-lam * positions_diff)
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