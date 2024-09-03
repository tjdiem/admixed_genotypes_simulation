from globals import *
from processing import *
from kmodels2 import KNet4
from math import log, e

@torch.no_grad()
def predict_cluster(model, SOI, positions_bp, recombination_map, len_chrom_bp=None, batch_size=batch_size, num_generations=None, admixture_proportion=None, population_size=None):
    #model.predict_cluster3

    if len_chrom_bp is None:
        len_chrom_bp = positions_bp[-1].item() * 2 - positions_bp[-2].item()

    len_chrom_morgan = recombination_map(len_chrom_bp)

    if num_generations is None:
        num_generations = 100 #### Change this default value
        infer_num_generations = True
    else:
        infer_num_generations = False

    if admixture_proportion is None:
        admixture_proportion = 0.5
        infer_admixture_proportion = True
    else:
        infer_admixture_proportion = False

    if population_size is None:
        population_size = 10_000

    # Then p_0a is the probability that the ancestry assigned class 0 correlates to ancestry A.
    # (meaning that it has ancesetry proportion 1 - AP)
    p_0a = 0.5
    p_2a = 1 - p_0a

    num_individuals, len_seq = SOI.shape

    padding = torch.full((num_individuals, input_size // 2), -1).to(device)
    SOI = torch.cat((padding, SOI, padding), dim=1) # (num_individuals, len_seq + input_size - 1)
    
    # fill with 0.25, 0.5, 0.25 # fill with ancestry proporotion
    predictions = torch.zeros((num_individuals, len_seq, num_classes)).to(device) # (num_individuals, len_seq, num_classes)

    positions_morgans = recombination_map(positions_bp)

    lam = 2 * population_size * (1 - e ** (-num_generations / (2*population_size)))
    lam_a = admixture_proportion * lam
    lam_c = (1 - admixture_proportion) * lam

    transition_aa_haploid = lam_c / lam + (lam_a/lam) * torch.exp(-lam * positions_morgans) # can make this more efficient by multiplying exps  # change in other locations too
    transition_cc_haploid = lam_a / lam + (lam_c/lam) * torch.exp(-lam * positions_morgans)

    prob_homozygous = 0.25 ########### (admixture_proportion ** 2 + (1 - admixture_proportion) ** 2) / 2
    prob_heterozygous = 0.5 ########### admixture_proportion * (1 - admixture_proportion) * 2
    # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    # 0.5 0.5 instead of 1 - ap, ap ??!!!
    infered_tract0 = 0.5 * (1 - admixture_proportion) * transition_aa_haploid + 0.5 * admixture_proportion * transition_cc_haploid
    infered_tract2 = 2 * prob_homozygous - infered_tract0
    noninfered_tract0 = (prob_homozygous * num_individuals - infered_tract0) / (num_individuals - 1)
    noninfered_tract2 = 2 * prob_homozygous - noninfered_tract0

    # predictions[...,[0,2]] = (admixture_proportion ** 2 + (1 - admixture_proportion) ** 2) / 2
    # predictions[...,1] = admixture_proportion * (1 - admixture_proportion) * 2

    predictions[0, :, 0] = infered_tract0
    predictions[0, :, 2] = infered_tract2
    predictions[1:, :, 0] = 0.25 #noninfered_tract0.unsqueeze(0).repeat(num_individuals - 1, 1)
    predictions[1:, :, 2] = 0.25 #noninfered_tract2.unsqueeze(0).repeat(num_individuals - 1, 1)
    predictions[:, :, 1] = 0.5

    # predictions = torch.rand_like(predictions)
    # predictions /= predictions.sum(dim=-1, keepdim=True)

    # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    # 0.5 0.5 instead of 1 - ap, ap ??!!!
    # predictions[0, :, 0] = 0.5 * (1 - admixture_proportion) * transition_aa_haploid + 0.5 * admixture_proportion * transition_cc_haploid
    # predictions[0, :, 2] = 0.5 - predictions[0, :, 0]

    padding = torch.full((num_individuals, input_size // 2, num_classes), 0).to(device)
    predictions = torch.cat((padding, predictions, padding), dim=1) # (num_individuals, len_seq + input_size - 1, num_classes)

    padding = torch.full((input_size // 2,), float("inf")).to(device)
    positions = torch.cat((padding, positions_morgans, padding), dim=0)

    mask = (1 - torch.eye(num_individuals)).bool() # is there some way we can make labels a pointer to predictions
    refs = SOI.unsqueeze(0).expand(num_individuals,-1,-1)[mask].reshape(num_individuals, num_individuals -1 , len_seq + input_size - 1)
    labels = predictions.unsqueeze(0).expand(num_individuals,-1,-1,-1)[mask].reshape(num_individuals, num_individuals - 1, len_seq + input_size - 1, num_classes)

    print("shapes")
    print(SOI.shape)
    print(refs.shape)
    print(predictions.shape)
    print(labels.shape)

    refs2 = torch.zeros(num_individuals, num_individuals - 1, len_seq + input_size - 1).to(device)
    for i in range(num_individuals):
        refs2[i] = torch.cat((SOI[:i], SOI[i+1:]), dim=0)
    print(torch.equal(refs, refs2))

    strict_predictions = predictions.clone()
        
    when_predicted = torch.full((num_individuals, len_seq), 1.0)
    has_predicted = torch.zeros((num_individuals, len_seq))
    # has_predicted[0, 0] = 1

    for i in range(0, num_individuals * len_seq * 10, batch_size):

        # probabilities = (when_predicted / when_predicted.sum()).flatten()
        # ind = torch.multinomial(probabilities, batch_size, replacement=False)
        # ind1 = ind // len_seq
        # ind2 = ind % len_seq

        # probabilities = (when_predicted / when_predicted.sum())

        probabilities = 1 - predictions[:, input_size // 2 : - (input_size // 2)].max(dim=-1)[0].cpu()
        if i == 0:
            probabilities[0] = 0
        ind1 = torch.multinomial(probabilities.sum(dim=-1), batch_size, replacement=False)
        ind2 = torch.multinomial(probabilities[ind1], 1).squeeze(-1)
        
        ind3, ind4 = ind1.clone(), ind2.clone() #############

        # ind1 = torch.randint(0, num_individuals, (batch_size,))
        # ind2 = torch.randint(0, len_seq, (batch_size,))

        has_predicted[ind1, ind2] = 1

        ind1 = ind1.unsqueeze(-1)  #faster way to index this?
        ind2 = ind2.unsqueeze(-1) + torch.arange(input_size).long()
        SOI_batch = SOI[ind1, ind2]
        refs_batch = refs[ind1, :, ind2].transpose(1,2)
        labels_batch = labels[ind1, :, ind2].transpose(1,2)
        positions_batch = positions.unsqueeze(0).expand(batch_size, -1)[torch.arange(batch_size).long().unsqueeze(-1), ind2]
        # torch.stack indices instead?  

        params_batch = torch.full((batch_size, 6), num_generations).to(device)

        out = model(SOI_batch, refs_batch, labels_batch, positions_batch, params_batch)
        out = F.softmax(out, dim=-1).double() # batch, num_classes
            
        ###########9999
        positions_diff = (positions - positions_batch[:, input_size // 2].unsqueeze(-1)).abs().to(device) # batch, len_seq + input_size

        transition_aa_haploid = lam_c / lam + (lam_a/lam) * torch.exp(-lam * positions_diff) # batch, len_seq + input_size - 1
        transition_cc_haploid = lam_a / lam + (lam_c/lam) * torch.exp(-lam * positions_diff)
        transition_ac_haploid = 1 - transition_aa_haploid
        transition_ca_haploid = 1 - transition_cc_haploid
        
        transitions = torch.zeros((batch_size, len_seq + input_size - 1, num_classes, num_classes)).double().to(device)
        transitions[:, :, 0, 0] = transition_aa_haploid ** 2
        transitions[:, :, 0, 1] = transition_aa_haploid * transition_ac_haploid * 2
        transitions[:, :, 0, 2] = transition_ac_haploid ** 2
        transitions[:, :, 1, 0] = transition_aa_haploid * transition_ca_haploid
        transitions[:, :, 1, 1] = transition_aa_haploid * transition_cc_haploid + transition_ac_haploid * transition_ca_haploid
        transitions[:, :, 1, 2] = transition_cc_haploid * transition_ac_haploid
        transitions[:, :, 2, 0] = transition_ca_haploid ** 2
        transitions[:, :, 2, 1] = transition_cc_haploid * transition_ca_haploid * 2
        transitions[:, :, 2, 2] = transition_cc_haploid ** 2

        out_smoothed = (out.unsqueeze(1).unsqueeze(1) @ transitions).squeeze(-2).float() #@ transition_ancestry_probs
        out = out.float()
        tmp = torch.exp(-2 * num_generations * positions_diff * 10).unsqueeze(-1) #hardcoded for now #increase factor as time goes on
        
        predictions[ind3] = predictions[ind3] * (1 - tmp) + out_smoothed * tmp
        predictions[:, :input_size // 2] = 0
        predictions[:, -(input_size // 2):] = 0

        assert ((1 - predictions[:, input_size // 2: - (input_size // 2)].sum(dim=-1)).abs() < 1e-4).all()
        ###########9999

        ###########9999
        strict_predictions[ind3, ind4] = out ###!!

        ###########9999

        ###########9999
        # conider multiplying exp distribution by alpha?
        # out = out.float()
        # len_exp_distribution = 49 ## this should be chosen based on num generations and threshold accuracy
        # positions_batch = (positions_batch[:, input_size // 2 - len_exp_distribution // 2: input_size // 2 + len_exp_distribution // 2 + 1] - positions_batch[:, input_size // 2].unsqueeze(-1)).abs()
        # # The below line is not exactly right.  Each predicted class could transition to another class with different probabilities
        # exp_distribution = torch.exp(-2 * num_generations * (positions_batch)) # batch, len_exp_distribution

        # out_smoothed = out.unsqueeze(1) * exp_distribution.unsqueeze(-1) # batch, len_exp_distribution, num_classes
        # predictions_idx = torch.stack([predictions[ind1[j,0], ind2[j,0] + input_size // 2 - len_exp_distribution // 2: ind2[j,0] + input_size // 2 + len_exp_distribution // 2 + 1] for j in range(batch_size)])
        # predictions_smoothed = predictions_idx * (1 - exp_distribution.unsqueeze(-1))

        # for j in range(batch_size):
        #     predictions[ind1[j,0], ind2[j,0] + input_size // 2 - len_exp_distribution // 2: ind2[j,0] + input_size // 2 + len_exp_distribution // 2 + 1] = out_smoothed[j] + predictions_smoothed[j]

        ###########9999

        labels = predictions.unsqueeze(0).expand(num_individuals,-1,-1,-1)[mask].reshape(num_individuals, num_individuals - 1, len_seq + input_size - 1, num_classes)

        if infer_num_generations and i % (batch_size * 500) == 0 and i > 0:

            predictions_argmax = predictions[:, input_size // 2: -(input_size // 2)].argmax(dim=-1)

            positions_diff = positions[input_size // 2: -(input_size // 2)]
            positions_diff_start = torch.tensor([(positions_diff[1] - positions_diff[0]) / 2 - 0]).to(device) # this assumes start morgans is 0
            positions_diff_end = torch.tensor([len_chrom_morgan - (positions_diff[-1] + positions_diff[-2]) / 2]).to(device)
            positions_diff = (positions_diff[2:] - positions_diff[:-2]) / 2
            positions_diff = torch.cat((positions_diff_start, positions_diff, positions_diff_end))
            positions_diff = positions_diff.unsqueeze(0).expand(num_individuals, -1)

            ####
            proportion_0 = positions_diff[predictions_argmax == 0].sum().item() / (len_chrom_morgan * num_individuals)
            proportion_1 = positions_diff[predictions_argmax == 1].sum().item() / (len_chrom_morgan * num_individuals) 
            proportion_2 = positions_diff[predictions_argmax == 2].sum().item() / (len_chrom_morgan * num_individuals)

            admixture_proportion = proportion_0 + proportion_1 * 0.5 + proportion_2 * 0
            ####

            #admixture_proportion = 1 - (predictions_argmax.sum().item() / (2 * len(chrom_morgan) * num_indiviudals))


            print(admixture_proportion)

            transitions = (predictions_argmax[:, :-1] - predictions_argmax[:, 1:])
            transitions_diff = transitions.sum().item()
            transitions_total = transitions.abs().sum().item() 
            num_transitions_10 = (transitions_total + transitions_diff) / 2
            num_transitions_01 = (transitions_total - transitions_diff) / 2

            num_transitions_10_per_morgan = num_transitions_10 / (2 * (1 - admixture_proportion) * len_chrom_morgan * num_individuals)
            num_transitions_01_per_morgan = num_transitions_01 / (2 * admixture_proportion * len_chrom_morgan * num_individuals)
            predicted_num_generations_10 = -2 * population_size * log(1 - num_transitions_10_per_morgan / (2 * population_size * admixture_proportion))
            predicted_num_generations_01 = -2 * population_size * log(1 - num_transitions_01_per_morgan / (2 * population_size * (1 - admixture_proportion)))
            
            print(predicted_num_generations_10)
            print(predicted_num_generations_01)
            print()

            # num_generations = (predicted_num_generations_01 + predicted_num_generations_10) / 2 

            admixture_proportion = 0.5

    return predictions[:, input_size // 2: -(input_size // 2)]
                                             
len_seq = 1500

random_file = int(sys.argv[1])

torch.manual_seed(409)
torch.manual_seed(202)
random.seed(202)

X, positions = convert_panel(panel_dir + "panel_" + str(random_file))
X = torch.tensor(X).to(device)[:49, :len_seq] # n_ind_adm, input_size
positions = torch.tensor(positions).to(device)[:len_seq]

y = convert_split(split_dir + "split_" + str(random_file), positions)
y = torch.tensor(y) # 2 * n_ind_adm, input_size
y = (y[::2] + y[1::2])[:49, :len_seq] # unphase ancestry labels # same shape as X

model = KNet4()
model = model.to(device)
model.load_state_dict(torch.load("cluster_13.pth", map_location=torch.device(device)))
model.eval()

num_bp = 50_000_000
recombination_map = lambda x: x / num_bp  # this assumes recombination rate is constant along chromosome, for chromosome of 50M base pairs
population_size = 10_000 # from looking at demography file. Should we vary this parameter?

with open(parameters_dir + "parameter_" + str(random_file)) as f:
    admixture_proportion, num_generations, *_ = f.readlines()
    admixture_proportion = float(admixture_proportion)
    num_generations = int(num_generations)

t1 = time.time()
y_pred = predict_cluster(model, X, positions, recombination_map=recombination_map, batch_size=16, num_generations=num_generations, admixture_proportion=None)
print(time.time() - t1)
y_pred = y_pred.argmax(dim=-1)
with open(f"cluster_pred/y_pred_{random_file}", "w") as f:
    f.write(str(y_pred.cpu().tolist()))
y = y.to(device)
for i in range(3):
    print((y_pred == i).sum().item())
    print((y == i).sum().item())
    print()

acc = max((y_pred == y).sum().item(), (2 - y_pred == y).sum().item()) / y.numel()
print(f"Accuracy: {acc:0.6f}")
