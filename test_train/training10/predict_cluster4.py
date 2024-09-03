from globals import *
from processing import *
from kmodels2 import KNet4
from math import log, e
from copy import deepcopy

@torch.no_grad()
def predict_cluster(model, SOI, positions_bp, recombination_map, len_chrom_bp=None, batch_size=batch_size, num_generations=None, admixture_proportion=None, population_size=None):

    torch.manual_seed(1111)
    if len_chrom_bp is None:
        len_chrom_bp = positions_bp[-1].item() * 2 - positions_bp[-2].item()

    if population_size is None:
        population_size = 10_000

    if admixture_proportion is None:
        admixture_proportion = 0.5

    len_chrom_morgan = recombination_map(len_chrom_bp)
    positions_morgans = recombination_map(positions_bp)

    num_individuals, len_seq = SOI.shape

    num_chunks = 10

    split_chunk_idx = torch.arange(num_chunks + 2).int() * len_seq // (num_chunks + 1) # should technically be based on position not index
    len_seq_chunked = split_chunk_idx[2:] - split_chunk_idx[:-2]
    SOI_chunked = [SOI[:, split_chunk_idx[i]:split_chunk_idx[i+2]] for i in range(num_chunks)] 

    positions_chunked = [positions_morgans[split_chunk_idx[i]:split_chunk_idx[i+2]] for i in range(num_chunks)]
    predictions_chunked = [torch.zeros((num_individuals, len_seq_chunked[i], num_classes)).to(device) for i in range(num_chunks)]

    refs_chunked = []
    labels_chunked = []

    len_seq_chunked_max = len_seq_chunked.max().item()
    padding_amount_chunked = len_seq_chunked - len_seq_chunked_max

    SOI_padding_chunked = [torch.full((num_individuals, padding_amount_chunked[chunk]), -1).to(device) for chunk in range(num_chunks)]
    SOI_chunked = [torch.cat((SOI_chunked[chunk], SOI_padding_chunked[chunk]), dim=1) for chunk in range(num_chunks)]
    SOI_chunked = torch.stack(SOI_chunked)

    print(SOI_chunked.shape)

    positions_padding_chunked = [torch.full((padding_amount_chunked[chunk],), float("inf")).to(device) for chunk in range(num_chunks)]
    positions_chunked = [torch.cat((positions_chunked[chunk], positions_padding_chunked[chunk]), dim=0) for chunk in range(num_chunks)]
    positions_chunked = torch.stack(positions_chunked)

    print(positions_chunked.shape)

    lam = 2 * population_size * (1 - e ** (-num_generations / (2*population_size)))
    lam_a = admixture_proportion * lam
    lam_c = (1 - admixture_proportion) * lam

    positions_morgans_tmp = (positions_chunked - positions_chunked[torch.arange(num_chunks).long(), len_seq_chunked // 2].unsqueeze(-1)).abs() # should technically be based on position not index
    transition_aa_haploid = lam_c / lam + (lam_a/lam) * torch.exp(-lam * positions_morgans_tmp) # can make this more efficient by multiplying exps  # change in other locations too
    transition_cc_haploid = lam_a / lam + (lam_c/lam) * torch.exp(-lam * positions_morgans_tmp)

    infered_tract0 = 0.5 * (1 - admixture_proportion) * transition_aa_haploid + 0.5 * admixture_proportion * transition_cc_haploid
    infered_tract2 = 0.5 - infered_tract0

    rand_index_chunked = torch.randint(num_individuals, (num_chunks,))
    predictions_chunked = torch.full((num_chunks, num_individuals, len_seq_chunked_max, num_classes), 0.25).to(device)
    predictions_chunked[..., 1] = 0.5
    predictions_chunked[torch.arange(num_chunks).long(), rand_index_chunked, :, 0] = infered_tract0
    predictions_chunked[torch.arange(num_chunks).long(), rand_index_chunked, :, 2] = infered_tract2

    
    padding = torch.full((num_chunks, num_individuals, input_size // 2), -1).to(device)
    SOI_chunked = torch.cat((padding, SOI_chunked, padding), dim=2)

    padding = torch.full((num_chunks, input_size // 2), float("inf")).to(device)
    positions_chunked = torch.cat((padding, positions_chunked, padding), dim=1)

    # Have seperate padded and unpadded predictions and make them pointers to each other
    # Same with other tensors
    padding = torch.full((num_chunks, num_individuals, input_size // 2, num_classes), 0).to(device)
    predictions_chunked = torch.cat((padding, predictions_chunked, padding), dim=2) # (num_individuals, len_seq + input_size - 1, num_classes)

    mask = (1 - torch.eye(num_individuals)).bool() # is there some way we can make labels a pointer to predictions
    refs_chunked = SOI_chunked.unsqueeze(1).expand(-1, num_individuals,-1,-1)[:, mask].reshape(num_chunks, num_individuals, num_individuals -1 , len_seq_chunked.max() + input_size - 1)
    labels_chunked = predictions_chunked.unsqueeze(1).expand(-1, num_individuals,-1,-1,-1)[:, mask].reshape(num_chunks, num_individuals, num_individuals - 1, len_seq_chunked.max() + input_size - 1, num_classes)

    print("jkfdljkdflkjd")
    print(SOI_chunked.shape)
    print(positions_chunked.shape)
    print(predictions_chunked.shape)
    print(refs_chunked.shape)
    print(labels_chunked.shape)

    if plotting:
        import matplotlib.pyplot as plt
        fig = plt.figure(figsize=(10,8))

    for i in range(0, num_individuals * len_seq * 2, batch_size):

        probabilities = 1 - 0 * predictions_chunked[:, :, input_size // 2 : - (input_size // 2)].max(dim=-1)[0].cpu()
        # if i == 0:
        #     probabilities[rand_index_chunked[chunk]] = 0
        ind12 = torch.multinomial(probabilities.sum(dim=-1).flatten(), batch_size, replacement=False)
        ind1 = ind12 // num_individuals
        ind2 = ind12 % num_individuals
        ind3 = torch.multinomial(probabilities[ind1, ind2], 1).squeeze(-1)
        
        expanded_ind1 = ind1.unsqueeze(-1)
        expanded_ind2 = ind2.unsqueeze(-1)
        expanded_ind3 = (ind3.unsqueeze(1) + torch.arange(input_size).long()).to(device)

        SOI_batch = SOI_chunked[expanded_ind1, expanded_ind2, expanded_ind3]    
        refs_batch = refs_chunked[expanded_ind1, expanded_ind2, :, expanded_ind3].transpose(1,2)
        labels_batch = labels_chunked[expanded_ind1, expanded_ind2, :, expanded_ind3].transpose(1,2)
        positions_batch = positions_chunked[expanded_ind1, expanded_ind3]
        

        ####!!
        # SOI_batch = torch.stack([SOI_chunked[ind1[j], ind2[j], ind3[j]:ind3[j] + input_size] for j in range(batch_size)])
        # refs_batch = torch.stack([refs_chunked[ind1[j], ind2[j], :, ind3[j]:ind3[j] + input_size] for j in range(batch_size)])
        # labels_batch = torch.stack([labels_chunked[ind1[j], ind2[j], :, ind3[j]:ind3[j] + input_size] for j in range(batch_size)])
        # positions_batch = torch.stack([positions_chunked[ind1[j], ind3[j]:ind3[j] + input_size] for j in range(batch_size)])
        ####!!


        params_batch = torch.full((batch_size, 6), num_generations).to(device) # fix this

        out = model(SOI_batch, refs_batch, labels_batch, positions_batch, params_batch)
        out = F.softmax(out, dim=-1).double() # batch, num_classes

        positions_diff = (positions_chunked[ind1] - positions_batch[:, input_size // 2].unsqueeze(-1)).abs().to(device) # batch, len_seq + input_size

        transition_aa_haploid = lam_c / lam + (lam_a/lam) * torch.exp(-lam * positions_diff) # batch, len_seq + input_size - 1
        transition_cc_haploid = lam_a / lam + (lam_c/lam) * torch.exp(-lam * positions_diff)
        transition_ac_haploid = 1 - transition_aa_haploid
        transition_ca_haploid = 1 - transition_cc_haploid
        
        transitions = torch.zeros((batch_size, len_seq_chunked_max + input_size - 1, num_classes, num_classes)).double().to(device)
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
        
        predictions_chunked[ind1, ind2] = predictions_chunked[ind1, ind2] * (1 - tmp) + out_smoothed * tmp
        predictions_chunked[:, :, :input_size // 2] = 0
        predictions_chunked[:, :, -(input_size // 2):] = 0 ##########################!!!!!!!!!!!!!!!!!!!!!!!!

        # assert ((1 - predictions_chunked[chunk][:, input_size // 2: - (input_size // 2)].sum(dim=-1)).abs() < 1e-4).all()

        labels_chunked = predictions_chunked.unsqueeze(1).expand(-1, num_individuals,-1,-1,-1)[:, mask].reshape(num_chunks, num_individuals, num_individuals - 1, len_seq_chunked.max() + input_size - 1, num_classes)

    # for chunk in range(num_chunks):
    #     torch.save(predictions_chunked[chunk], f"tempdir/predictions_chunked{chunk}.pt")
        
        # import matplotlib.pyplot as plt

        # fig = plt.figure(figsize=(10,8))
        # for chunk in range(num_chunks):
        #     left = chunk / (num_chunks + 1)
        #     bottom = 1 - (chunk + 1) / num_chunks
        #     width = 2 / (num_chunks + 1)
        #     height = 1 / num_chunks
        #     ax = fig.add_axes([left, bottom, width, height])
        #     ax.imshow(predictions_chunked[chunk][:, input_size // 2: - (input_size // 2)].cpu(), interpolation="nearest", aspect="auto")
        
        # plt.tight_layout()
        # plt.show()

        # for chunk in range(num_chunks - 1):

            # left_region = predictions_chunked[chunk][:, input_size // 2 + split_chunk_idx[chunk + 1] - split_chunk_idx[chunk]: -(input_size // 2)]
            # right_region = predictions_chunked[chunk + 1][:, input_size // 2: split_chunk_idx[chunk + 2] - split_chunk_idx[chunk + 1] + (input_size // 2)]

            # diff_same = (left_region - right_region).abs().sum()
            # diff_switch = (left_region - right_region[...,[2,1,0]]).abs().sum()

            # if diff_same > diff_switch:
            #     predictions_chunked[chunk + 1] = predictions_chunked[chunk + 1][...,[2,1,0]]

            # input_size // 2 + len_seq_max // 2 + len_seq_max // 2 // 2

        left_strip = predictions_chunked[:-1, :, input_size // 2 + 3 * len_seq_chunked_max // 4]
        right_strip = predictions_chunked[1:, :, input_size // 2 + len_seq_chunked_max // 4]

        # print(left_strip[0])
        # print(right_strip[0])

        prob_same = (left_strip * right_strip).sum(dim=-1).prod(dim=-1)
        prob_switch = (left_strip * right_strip[...,[2,1,0]]).sum(dim=-1).prod(dim=-1)

        prob_same /= prob_same + prob_switch

        # print(prob_same)


        if plotting:

            fig.clear()
            for chunk in range(num_chunks):
                left = chunk / (num_chunks + 1)
                bottom = 1 - (chunk + 1) / num_chunks
                width = 2 / (num_chunks + 1)
                height = 1 / num_chunks
                ax = fig.add_axes([left, bottom, width, height])
                ax.imshow(predictions_chunked[chunk][:, input_size // 2: - (input_size // 2)].cpu(), interpolation="nearest", aspect="auto")
            
            for chunk in range(num_chunks - 1):
                fig.text((chunk + 0.5) / (num_chunks + 1), 1 - (chunk + 1.5) / num_chunks, f"{prob_same[chunk].item():0.5f}", ha="center", va="center", fontsize=12)

            plt.tight_layout()
            plt.pause(0.05)
            # plt.show()

    switched = 1 - prob_same.round().int()
    switched = (torch.cumsum(switched, 0) % 2).bool()
    predictions_chunked[1:][switched] = predictions_chunked[1:][switched][...,[2,1,0]]
    predictions = torch.zeros((num_individuals, len_seq, num_classes)).to(device)
    for chunk in range(num_chunks):
        predictions[:, split_chunk_idx[chunk]: split_chunk_idx[chunk + 2]] += predictions_chunked[chunk][:, input_size // 2: -(input_size // 2)]

    predictions /= predictions.sum(dim=-1, keepdim=True)
    predictions_prev = predictions.clone()

    padding = torch.full((num_individuals, input_size // 2), -1).to(device)
    SOI = torch.cat((padding, SOI, padding), dim=1) # (num_individuals, len_seq + input_size - 1)

    padding = torch.full((num_individuals, input_size // 2, num_classes), 0).to(device)
    predictions = torch.cat((padding, predictions, padding), dim=1) # (num_individuals, len_seq + input_size - 1, num_classes)

    padding = torch.full((input_size // 2,), float("inf")).to(device)
    positions = torch.cat((padding, positions_morgans, padding), dim=0)

    mask = (1 - torch.eye(num_individuals)).bool() # is there some way we can make labels a pointer to predictions
    refs = SOI.unsqueeze(0).expand(num_individuals,-1,-1)[mask].reshape(num_individuals, num_individuals -1 , len_seq + input_size - 1)
    labels = predictions.unsqueeze(0).expand(num_individuals,-1,-1,-1)[mask].reshape(num_individuals, num_individuals - 1, len_seq + input_size - 1, num_classes)

    for i in range(0, num_individuals * len_seq, batch_size):

        probabilities = 1 - predictions[:, input_size // 2 : - (input_size // 2)].max(dim=-1)[0].cpu()
        if i == 0:
            probabilities[0] = 0
        ind1 = torch.multinomial(probabilities.sum(dim=-1), batch_size, replacement=False)
        ind2 = torch.multinomial(probabilities[ind1], 1).squeeze(-1)
        
        ind3, ind4 = ind1.clone(), ind2.clone() #############

        ind1 = ind1.unsqueeze(-1)  #faster way to index this?
        ind2 = ind2.unsqueeze(-1) + torch.arange(input_size).long()
        SOI_batch = SOI[ind1, ind2]
        refs_batch = refs[ind1, :, ind2].transpose(1,2)
        labels_batch = labels[ind1, :, ind2].transpose(1,2)
        positions_batch = positions.unsqueeze(0).expand(batch_size, -1)[torch.arange(batch_size).long().unsqueeze(-1), ind2]

        params_batch = torch.full((batch_size, 6), num_generations).to(device)

        out = model(SOI_batch, refs_batch, labels_batch, positions_batch, params_batch)
        out = F.softmax(out, dim=-1).double() # batch, num_classes
            
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

        labels = predictions.unsqueeze(0).expand(num_individuals,-1,-1,-1)[mask].reshape(num_individuals, num_individuals - 1, len_seq + input_size - 1, num_classes)


    return predictions_prev, predictions[:, input_size // 2: -(input_size // 2)]

plotting = False
                                             
len_seq = 2750

random_file = int(sys.argv[1])

torch.manual_seed(409)
# random.seed(12345)

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
y_pred_prev, y_pred = predict_cluster(model, X, positions, recombination_map=recombination_map, batch_size=16, num_generations=num_generations, admixture_proportion=None)
print(time.time() - t1)
y_pred = y_pred.argmax(dim=-1)
with open(f"cluster_pred/y_pred_{random_file}", "w") as f:
    f.write(str(y_pred.cpu().tolist()))
y = y.to(device)
for i in range(3):
    print((y_pred == i).sum().item())
    print((y == i).sum().item())
    print()

# true values vary left to right
CM = torch.zeros(3, 3)
for i in range(3):
    for j in range(3):
        CM[i, j] = ((y_pred == i) & (y == j)).sum().item()

print(CM)
print()

acc = max((y_pred == y).sum().item(), (2 - y_pred == y).sum().item()) / y.numel()
print(f"Accuracy: {acc:0.6f}")

y_pred = y_pred_prev.argmax(dim=-1)
acc = max((y_pred == y).sum().item(), (2 - y_pred == y).sum().item()) / y.numel()
print(f"Accuracy prev: {acc:0.6f}")

