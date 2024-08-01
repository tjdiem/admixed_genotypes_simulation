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
    predictions_weights = torch.zeros((len_seq, 1)).to(device)

    predicted_idx = torch.zeros((len_seq)).bool()

    positions_next = torch.arange(batch_size).to(device) * ((end_pos - start_pos)/ batch_size) + start_pos + 1/(2*batch_size)
    positions_diff = (positions - positions_next.unsqueeze(-1)).abs() # batch, len_seq
    idx_next = torch.argmin(positions_diff, dim=-1)  #faster way to do this since positions is already sorted?
    # start - ((pos - start) * 1/3 )
    # 4/3 * start - 1/3 * pos

    # end + ((end - pos) * 1/3)
    # 4/3 * end - 1/3 * pos

    num_iterations = 20
    k1 = 22
    k2 = 20
    outs = torch.zeros((num_iterations * batch_size, 3)).float().to(device)
    idx_sampled = torch.arange(batch_size * num_iterations).long() * len_seq // (batch_size * num_iterations)
    # idx_sampled = torch.randperm(len_seq)[:num_iterations * batch_size].sort()[0]
    for iteration in range(num_iterations):

        ##!!
        # iteration = num_iterations - iteration - 1
        start = iteration * len_seq / num_iterations
        end = (iteration + 1) * len_seq / num_iterations
        # idx_next = (torch.arange(batch_size) * (end - start) / batch_size + start).long()
        idx_next = idx_sampled[iteration * batch_size: (iteration + 1) * batch_size] 
        ##!!

        predicted_idx[idx_next] = 1
        predicted_pos = positions[predicted_idx]
        predicted_pos = torch.cat(((4/3 * start_pos - 1/3 * predicted_pos[0]).unsqueeze(0), predicted_pos, (4/3 * end_pos - 1/3 * predicted_pos[-1]).unsqueeze(0)), dim=0)

    
        # faster way to index this?
        SOI_batch = torch.stack([SOI_padded[idx_next[i]:idx_next[i]+input_size] for i in range(batch_size)])
        positions_batch = torch.stack([positions_padded[idx_next[i]:idx_next[i]+input_size] for i in range(batch_size)])
        refs_batch = torch.stack([refs_padded[:, idx_next[i]:idx_next[i]+input_size] for i in range(batch_size)])
        labels_batch = torch.stack([labels_padded[:, idx_next[i]:idx_next[i]+input_size] for i in range(batch_size)])
        params_batch = torch.full((batch_size, 6), num_generations).to(device)  # this is messy
        
        out = model(SOI_batch, refs_batch, labels_batch, positions_batch, params_batch)
        out = F.softmax(out, dim=-1).double() # batch, num_classes

        # positions_diff = (positions - positions_batch[:, input_size // 2].unsqueeze(-1)).abs() # batch, len_seq + input_size

        # # below can be simplified by a lot if we are assuming admixture proportion of 0.5
        # transition_aa_haploid = lam_c / lam + (lam_a/lam) * torch.exp(-lam * positions_diff) # batch, len_seq + input_size - 1
        # transition_cc_haploid = lam_a / lam + (lam_c/lam) * torch.exp(-lam * positions_diff)
        # transition_ac_haploid = 1 - transition_aa_haploid
        # transition_ca_haploid = 1 - transition_cc_haploid
        
        # transitions = torch.zeros((batch_size, len_seq, num_classes, num_classes)).double().to(device)
        # transitions[:, :, 0, 0] = transition_aa_haploid ** 2
        # transitions[:, :, 0, 1] = transition_aa_haploid * transition_ac_haploid * 2
        # transitions[:, :, 0, 2] = transition_ac_haploid ** 2
        # transitions[:, :, 1, 0] = transition_aa_haploid * transition_ca_haploid
        # transitions[:, :, 1, 1] = transition_aa_haploid * transition_cc_haploid + transition_ac_haploid * transition_ca_haploid
        # transitions[:, :, 1, 2] = transition_cc_haploid * transition_ac_haploid
        # transitions[:, :, 2, 0] = transition_ca_haploid ** 2
        # transitions[:, :, 2, 1] = transition_cc_haploid * transition_ca_haploid * 2
        # transitions[:, :, 2, 2] = transition_cc_haploid ** 2

        outs[iteration * batch_size: (iteration + 1) * batch_size] = out.float()
        # out_smoothed = (out.unsqueeze(1).unsqueeze(1) @ transitions).squeeze(-2).float() #@ transition_ancestry_probs
        # out = out.float()
        # position_weights = torch.exp(-k * num_generations * positions_diff).unsqueeze(-1) #hardcoded for now #increase factor as time goes on
        
        # predictions += (out_smoothed * position_weights).sum(dim=0)
        # predictions_weights += position_weights.sum(dim=0)
        
        # for a in range(3):
        #     plt.plot(torch.arange(len_seq).int(), predictions[:,a].cpu(),label=f"ancestry {a}")
        # plt.show()

        # fig, (ax1, ax2) = plt.subplots(1, 2)
        # ax1.imshow((predictions/predictions_weights).unsqueeze(-1).expand(-1,-1,1000).reshape(-1,3000).cpu())
        # ax2.imshow(F.one_hot(y, num_classes=3).unsqueeze(-1).expand(-1,-1,1000).reshape(-1,3000).cpu())
        # plt.show()

        # plt.scatter(positions.cpu(), predicted_idx.cpu())
        # plt.show()

        # for a in range(batch_size):
        #     plt.imshow(out_smoothed[a].unsqueeze(-1).expand(-1,-1,1000).reshape(-1,3000).cpu())
        #     plt.colorbar()
        #     plt.show()



        ##!!
        # gap_lengths = predicted_pos[1:] - predicted_pos[:-1]
        # gap_lengths, gap_idx = torch.topk(gap_lengths, batch_size, largest=True)
        # positions_next = predicted_pos[gap_idx] + gap_lengths / 2
        # positions_diff = (positions - positions_next.unsqueeze(-1)).abs() # batch, len_seq
        # idx_next = torch.argmin(positions_diff, dim=-1)  #faster way to do this since positions is already sorted?
        ##!!

        ##!!
        #torch.cumsum and torch_scatter.scatter_min()
        # predictions_probs = predictions.max(dim=-1)[0]
        # predicted_idx_where = torch.nonzero(predicted_idx).squeeze(-1)
        # if predicted_idx_where[0] == 0:
        #     predicted_idx_where = torch.cat((torch.nonzero(predicted_idx).squeeze(-1), torch.tensor([len_seq]))).to(device)
        # else:
        #     predicted_idx_where = torch.cat((torch.tensor([0]), torch.nonzero(predicted_idx).squeeze(-1), torch.tensor([len_seq]))).to(device)
        # sections = torch.split(predictions_probs, (predicted_idx_where[1:] - predicted_idx_where[:-1]).tolist())
        # min_sections_idx = torch.stack([torch.argmin(section) for section in sections])
        # min_sections_idx = predicted_idx_where[:-1] + min_sections_idx
        # _, min_sections_idx_idx = torch.topk(predictions_probs[min_sections_idx], batch_size, largest=False)
        # idx_next = min_sections_idx[min_sections_idx_idx]
        ##!!


        # ##!!
        # # iteration = 19 - iteration
        # start = iteration * len_seq / 20
        # end = (iteration + 1) * len_seq / 20
        # idx_next = (torch.arange(batch_size) * (end - start) / batch_size + start).long()
        # print(idx_next)
        # ##!!

    # outs (num_predictions, 3)
    positions_predicted = positions[predicted_idx] # (num_predictions)

    # positions_diff = (positions - positions_predicted.unsqueeze(-1)).abs() # batch, len_seq + input_size

    # # below can be simplified by a lot if we are assuming admixture proportion of 0.5
    # transition_aa_haploid = lam_c / lam + (lam_a/lam) * torch.exp(-lam * positions_diff) # batch, len_seq + input_size - 1
    # transition_cc_haploid = lam_a / lam + (lam_c/lam) * torch.exp(-lam * positions_diff)
    # transition_ac_haploid = 1 - transition_aa_haploid
    # transition_ca_haploid = 1 - transition_cc_haploid
    
    # transitions = torch.zeros((batch_size * num_iterations, len_seq, num_classes, num_classes)).float().to(device)
    # transitions[:, :, 0, 0] = transition_aa_haploid ** 2
    # transitions[:, :, 0, 1] = transition_aa_haploid * transition_ac_haploid * 2
    # transitions[:, :, 0, 2] = transition_ac_haploid ** 2
    # transitions[:, :, 1, 0] = transition_aa_haploid * transition_ca_haploid
    # transitions[:, :, 1, 1] = transition_aa_haploid * transition_cc_haploid + transition_ac_haploid * transition_ca_haploid
    # transitions[:, :, 1, 2] = transition_cc_haploid * transition_ac_haploid
    # transitions[:, :, 2, 0] = transition_ca_haploid ** 2
    # transitions[:, :, 2, 1] = transition_cc_haploid * transition_ca_haploid * 2
    # transitions[:, :, 2, 2] = transition_cc_haploid ** 2

    # transitions[:, :, 0, 0] = 1
    # transitions[:, :, 0, 1] = 0
    # transitions[:, :, 0, 2] = 0
    # transitions[:, :, 1, 0] = 0
    # transitions[:, :, 1, 1] = 1
    # transitions[:, :, 1, 2] = 0
    # transitions[:, :, 2, 0] = 0
    # transitions[:, :, 2, 1] = 0
    # transitions[:, :, 2, 2] = 1

    # predictions = (transitions * outs.unsqueeze(1).unsqueeze(1)).sum(dim=-1)
    # predictions = predictions.log().sum(dim=0)

    # positions_diff = torch.cat((torch.tensor([0]).to(device), positions_predicted[1:] - positions_predicted[:-1])) # batch, len_seq + input_size
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

        if i % 100 == 0:
            print(i)
        
            
        
    # for i in range(outs.shape[0]):
    #     print(outs[i])
    #     print(cum_probs_right[i])
    #     print(cum_probs_left[i])
    #     print()

    # density = torch.exp(-k2 * num_generations * (positions_predicted - positions_predicted.unsqueeze(-1)).abs())
    # area = 2 - torch.exp(-k2 * num_generations * (positions_predicted - start_pos)) - torch.exp(-k2 * num_generations * (end_pos - positions_predicted))
    # density_weights = area / density.sum(dim=-1)
    # density_weights /= density_weights.sum()

    # print(torch.nonzero(predicted_idx))
    # outs *= density_weights.unsqueeze(-1)


    # out_smoothed = (outs.unsqueeze(1).unsqueeze(1) @ transitions).squeeze(-2).float() #@ transition_ancestry_probs
    # position_weights = torch.exp(-k1 * lam * positions_diff).unsqueeze(-1) #hardcoded for now #increase factor as time goes on
    
    # print(density_weights[:20])

    # plt.scatter(positions_predicted[:20].cpu(), torch.ones(20,))
    # plt.show()

    
    # predictions += (out_smoothed * position_weights).sum(dim=0)
    # predictions_weights += position_weights.sum(dim=0)

    # predictions /= predictions_weights

    return predictions


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

# idx_correct = (y_pred_strict == y).cpu()

# plt.hist(y_pred[idx_correct].amax(dim=-1).cpu())
# plt.hist(y_pred[~idx_correct].amax(dim=-1).cpu())

# fig, (ax1, ax2) = plt.subplots(1, 2)
# ax1.hist(y_pred[idx_correct].amax(dim=-1).cpu())
# ax2.hist(y_pred[~idx_correct].amax(dim=-1).cpu())
# plt.show()

# num_ind = 190000
# num_ind = min(num_ind, y_pred.shape[0])
# plt.scatter(torch.arange(num_ind), y_pred[:num_ind].amax(dim=-1).cpu())
# plt.scatter(torch.arange(y_pred.shape[0])[:num_ind][~idx_correct[:num_ind]], torch.zeros((num_ind - idx_correct[:num_ind].sum())))
# idx_sampled = torch.arange(batch_size * 20).long() * y_pred.shape[0] // (batch_size * 20)
# plt.scatter(idx_sampled, torch.full((batch_size * 20,), 0.25))
# plt.show()