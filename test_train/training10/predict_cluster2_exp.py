from globals import *
from processing import *
from kmodels2 import KNet4
from math import log, e
from copy import deepcopy

"""
ways to solve the random seed problem:
figure out if there is any pattern in the seeds that have bad vs good results
run multiple iterations and cut off ones with low avg certainty
run multiple iterations and average the results
figure out better way to choose next indexes
problem might solve itself once we train on clustering examples
"""


@torch.no_grad()
def predict_cluster(model, SOI, positions_bp, recombination_map, len_chrom_bp=None, batch_size=batch_size, num_generations=None, admixture_proportion=None, population_size=None):

    if len_chrom_bp is None:
        len_chrom_bp = positions_bp[-1].item() * 2 - positions_bp[-2].item()

    if population_size is None:
        population_size = 10_000

    if admixture_proportion is None:
        admixture_proportion = 0.5

    len_chrom_morgan = recombination_map(len_chrom_bp)
    positions_morgans = recombination_map(positions_bp)

    num_individuals, len_seq = SOI.shape

    num_chunks = 1

    split_chunk_idx = torch.arange(num_chunks + 2).int() * len_seq // (num_chunks + 1) # should technically be based on position not index
    len_seq_chunked = split_chunk_idx[2:] - split_chunk_idx[:-2]
    SOI_chunked = [SOI[:, split_chunk_idx[i]:split_chunk_idx[i+2]] for i in range(num_chunks)]

    positions_chunked = [positions_morgans[split_chunk_idx[i]:split_chunk_idx[i+2]] for i in range(num_chunks)]
    predictions_chunked = [torch.zeros((num_individuals, len_seq_chunked[i], num_classes)).to(device) for i in range(num_chunks)]

    refs_chunked = []
    labels_chunked = []

    rand_index_chunked = [] # ???????
    for chunk in range(num_chunks):

        # masks and padding within this for loop can be reused
        # some lines can be written as list comprehension and parallelized

        padding = torch.full((num_individuals, input_size // 2), -1).to(device)
        SOI_chunked[chunk] = torch.cat((padding, SOI_chunked[chunk], padding), dim=1) # (num_individuals, len_seq + input_size - 1)
        
        lam = 2 * population_size * (1 - e ** (-num_generations / (2*population_size)))
        lam_a = admixture_proportion * lam
        lam_c = (1 - admixture_proportion) * lam

        positions_morgans_tmp = (positions_chunked[chunk] - positions_chunked[chunk][len_seq_chunked[chunk] // 2]).abs() # should technically be based on position not index
        transition_aa_haploid = lam_c / lam + (lam_a/lam) * torch.exp(-lam * positions_morgans_tmp) # can make this more efficient by multiplying exps  # change in other locations too
        transition_cc_haploid = lam_a / lam + (lam_c/lam) * torch.exp(-lam * positions_morgans_tmp)

        infered_tract0 = 0.5 * (1 - admixture_proportion) * transition_aa_haploid + 0.5 * admixture_proportion * transition_cc_haploid
        infered_tract2 = 0.5 - infered_tract0

            
        rand_index = random.randint(0, num_individuals - 1) # try doing this with the same index for each individual
        rand_index = int(sys.argv[3])   
        predictions_chunked[chunk][:, :, 0] = 0.25 
        predictions_chunked[chunk][:, :, 2] = 0.25
        predictions_chunked[chunk][:, :, 1] = 0.5
        predictions_chunked[chunk][rand_index, :, 0] = infered_tract0
        predictions_chunked[chunk][rand_index, :, 2] = infered_tract2

        rand_index_chunked.append(rand_index)

        padding = torch.full((num_individuals, input_size // 2, num_classes), 0).to(device)
        predictions_chunked[chunk] = torch.cat((padding, predictions_chunked[chunk], padding), dim=1) # (num_individuals, len_seq + input_size - 1, num_classes)

        padding = torch.full((input_size // 2,), float("inf")).to(device)
        positions_chunked[chunk] = torch.cat((padding, positions_chunked[chunk], padding), dim=0)

        mask = (1 - torch.eye(num_individuals)).bool() # is there some way we can make labels a pointer to predictions
        refs_chunked.append(SOI_chunked[chunk].unsqueeze(0).expand(num_individuals,-1,-1)[mask].reshape(num_individuals, num_individuals -1 , len_seq_chunked[chunk] + input_size - 1))
        labels_chunked.append(predictions_chunked[chunk].unsqueeze(0).expand(num_individuals,-1,-1,-1)[mask].reshape(num_individuals, num_individuals - 1, len_seq_chunked[chunk] + input_size - 1, num_classes))

    for chunk in range(0, num_chunks):

        # if chunk == 1:
        #     torch.manual_seed(202)
        for i in range(0, num_individuals * len_seq_chunked[chunk], batch_size):

            probabilities = 1 - predictions_chunked[chunk][:, input_size // 2 : - (input_size // 2)].max(dim=-1)[0].cpu()
            if i == 0:
                probabilities[rand_index_chunked[chunk]] = 0
            ind1 = torch.multinomial(probabilities.sum(dim=-1), batch_size, replacement=False)
            ind2 = torch.multinomial(probabilities[ind1], 1).squeeze(-1)
            
            ind3, ind4 = ind1.clone(), ind2.clone() #############


            ind1 = ind1.unsqueeze(-1)  #faster way to index this?
            ind2 = ind2.unsqueeze(-1) + torch.arange(input_size).long()
            SOI_batch = SOI_chunked[chunk][ind1, ind2]
            refs_batch = refs_chunked[chunk][ind1, :, ind2].transpose(1,2)
            labels_batch = labels_chunked[chunk][ind1, :, ind2].transpose(1,2)
            positions_batch = positions_chunked[chunk].unsqueeze(0).expand(batch_size, -1)[torch.arange(batch_size).long().unsqueeze(-1), ind2]

            params_batch = torch.full((batch_size, 6), num_generations).to(device)

            out = model(SOI_batch, refs_batch, labels_batch, positions_batch, params_batch)
            out = F.softmax(out, dim=-1).double() # batch, num_classes

            # if i < 4 * batch_size:
            #     print(chunk)
            #     print(ind3)
            #     print(ind4)
            #     print((out.argmax(dim=-1) == y.to(device)[ind3, ind4]).sum())
            #     print()

            positions_diff = (positions_chunked[chunk] - positions_batch[:, input_size // 2].unsqueeze(-1)).abs().to(device) # batch, len_seq + input_size

            transition_aa_haploid = lam_c / lam + (lam_a/lam) * torch.exp(-lam * positions_diff) # batch, len_seq + input_size - 1
            transition_cc_haploid = lam_a / lam + (lam_c/lam) * torch.exp(-lam * positions_diff)
            transition_ac_haploid = 1 - transition_aa_haploid
            transition_ca_haploid = 1 - transition_cc_haploid
            
            transitions = torch.zeros((batch_size, len_seq_chunked[chunk] + input_size - 1, num_classes, num_classes)).double().to(device)
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
            
            predictions_chunked[chunk][ind3] = predictions_chunked[chunk][ind3] * (1 - tmp) + out_smoothed * tmp
            predictions_chunked[chunk][:, :input_size // 2] = 0
            predictions_chunked[chunk][:, -(input_size // 2):] = 0

            assert ((1 - predictions_chunked[chunk][:, input_size // 2: - (input_size // 2)].sum(dim=-1)).abs() < 1e-4).all()

            labels_chunked[chunk] = predictions_chunked[chunk].unsqueeze(0).expand(num_individuals,-1,-1,-1)[mask].reshape(num_individuals, num_individuals - 1, len_seq_chunked[chunk] + input_size - 1, num_classes)

    if False:
        import matplotlib.pyplot as plt

        fig = plt.figure(figsize=(10,8))
        for chunk in range(num_chunks):
            left = chunk / (num_chunks + 1)
            bottom = 1 - (chunk + 1) / num_chunks
            width = 2 / (num_chunks + 1)
            height = 1 / num_chunks
            ax = fig.add_axes([left, bottom, width, height])
            ax.imshow(predictions_chunked[chunk][:, input_size // 2: - (input_size // 2)].cpu(), interpolation="nearest", aspect="auto")
        
        plt.tight_layout()
        plt.show()

    # for chunk in range(num_chunks):
    #     torch.save(predictions_chunked[chunk], f"tempdir/predictions_chunked{chunk}.pt")

    for chunk in range(num_chunks - 1):
        left_region = predictions_chunked[chunk][:, input_size // 2 + split_chunk_idx[chunk + 1] - split_chunk_idx[chunk]: -(input_size // 2)]
        right_region = predictions_chunked[chunk + 1][:, input_size // 2: split_chunk_idx[chunk + 2] - split_chunk_idx[chunk + 1] + (input_size // 2)]


        diff_same = (left_region - right_region).abs().sum()
        diff_switch = (left_region - right_region[...,[2,1,0]]).abs().sum()

        if diff_same > diff_switch:
            predictions_chunked[chunk + 1] = predictions_chunked[chunk + 1][...,[2,1,0]]

    predictions = torch.zeros((num_individuals, len_seq, num_classes)).to(device)
    for chunk in range(num_chunks):
        predictions[:, split_chunk_idx[chunk]: split_chunk_idx[chunk + 2]] += predictions_chunked[chunk][:, input_size // 2: -(input_size // 2)]


    if False:
        mask = (1 - torch.eye(num_individuals)).bool() # is there some way we can make labels a pointer to predictions
        refs = SOI.unsqueeze(0).expand(num_individuals,-1,-1)[mask].reshape(num_individuals, num_individuals -1 , input_size)
        labels = predictions.unsqueeze(0).expand(num_individuals,-1,-1,-1)[mask].reshape(num_individuals, num_individuals - 1, input_size, num_classes)

        labels = F.one_hot(labels.argmax(dim=-1), num_classes=num_classes)

        params = torch.full((1, 6), num_generations).to(device)

        predictions_strict = F.one_hot(predictions.argmax(dim=-1), num_classes=num_classes)

        WCSS = 0
        for i in range(49):
            out = model(SOI[i].unsqueeze(0), refs[i].unsqueeze(0), labels[i].unsqueeze(0), positions_morgans.unsqueeze(0), params)
            # print(SOI[i].unsqueeze(0), refs[i].unsqueeze(0), labels[i].unsqueeze(0), positions_morgans.unsqueeze(0), params)
            out = F.softmax(out, dim=-1)
            WCSS += (out * predictions_strict[i, 250]).sum().item()

        WCSS /= 49
        print()
        print("WCSS: ", WCSS)

    if False:
        padding = torch.full((num_individuals, input_size // 2), -1).to(device)
        SOI = torch.cat((padding, SOI, padding), dim=1) # (num_individuals, len_seq + input_size - 1)

        padding = torch.full((num_individuals, input_size // 2, num_classes), 0).to(device)
        predictions = torch.cat((padding, predictions, padding), dim=1) # (num_individuals, len_seq + input_size - 1, num_classes)

        padding = torch.full((input_size // 2,), float("inf")).to(device)
        positions = torch.cat((padding, positions_morgans, padding), dim=0)

        mask = (1 - torch.eye(num_individuals)).bool() # is there some way we can make labels a pointer to predictions
        refs = SOI.unsqueeze(0).expand(num_individuals,-1,-1)[mask].reshape(num_individuals, num_individuals -1 , input_size + len_seq - 1)
        labels = predictions.unsqueeze(0).expand(num_individuals,-1,-1,-1)[mask].reshape(num_individuals, num_individuals - 1, input_size + len_seq - 1, num_classes)

        labels = F.one_hot(labels.argmax(dim=-1), num_classes=num_classes)
        labels[:, :, :input_size // 2] = 0
        labels[:, :, -(input_size // 2):] = 0

        params = torch.full((1, 6), num_generations).to(device)

        predictions_strict = F.one_hot(predictions.argmax(dim=-1), num_classes=num_classes)

        WCSS = 0
        for i in range(700):
            SOI_batch = SOI[random_idx_test1[i], random_idx_test2[i]:random_idx_test2[i] + input_size].unsqueeze(0)
            refs_batch = refs[random_idx_test1[i], :, random_idx_test2[i]:random_idx_test2[i] + input_size].unsqueeze(0)
            labels_batch = labels[random_idx_test1[i], :,  random_idx_test2[i]:random_idx_test2[i] + input_size].unsqueeze(0)
            positions_batch = positions[random_idx_test2[i]:random_idx_test2[i] + input_size]

            mask = torch.ones_like(refs).bool().to(device)
            mask[:, random_idx_test3[i]] = 0

            labels[mask] = 0
            refs[mask] = -1

            out = model(SOI_batch, refs_batch, labels_batch, positions_morgans.unsqueeze(0), params)
            # print(SOI[i].unsqueeze(0), refs[i].unsqueeze(0), labels[i].unsqueeze(0), positions_morgans.unsqueeze(0), params)
            out = F.softmax(out, dim=-1)
            WCSS += (out * predictions_strict[random_idx_test1[i], random_idx_test2[i] + 250]).sum().item()

        WCSS /= 700
        print()
        print("Full WCSS: ", WCSS)

        predictions = predictions[:, 250: -250]

    if False:
        total_dist = 0
        total_ex = 0

        strict_predictions = predictions.argmax(dim=-1)

        for i in range(700):
            ind1 = random_idx_test1[i].item()
            ind2 = random_idx_test2[i]
            ind3 = (ind1 + random_idx_test3[i] + 1) % 49

            if (strict_predictions[ind1, ind2 + 200: ind2 + 301] == strict_predictions[ind3, ind2 + 200: ind2 + 301]).sum().item() > 80:
                total_dist += (SOI[ind1, ind2 + 200: ind2 + 301] != SOI[ind3, ind2 + 200: ind2 + 301]).sum().item()
                total_ex += 1

        print("num_examples: ", total_ex)
        print("avg_dist: ", total_dist / total_ex)

    if False:
        with open(parameters_dir + "parameter_" + str(random_file)) as f:
            ap = float(f.readlines()[0])

        class_probs1 = torch.tensor([ap ** 2, 2 * ap * (1 - ap), (1 - ap) ** 2]).to(device)
        class_probs2 = class_probs1[[2,1,0]]

        samples = predictions[:, 250: -250][:, ::100]

        strict_samples = F.one_hot(samples.argmax(dim=-1), num_classes=num_classes)

        log_prob = max((samples * class_probs1).log().sum().item(), (samples * class_probs2).log().sum().item())

        strict_log_prob = max((strict_samples * class_probs1).log().sum().item(), (samples * class_probs2).log().sum().item())

        print(f"prob ap: exp({log_prob})")
        print(f"strict prob ap: exp({strict_log_prob})")

    return predictions

# sys: 
# 1: file number 
# 2: random seed
# 3: inferred individual
# 4: start seq index

start_seq = int(sys.argv[4])
len_seq = 500

random_idx_test1 = torch.randint(0, 49, (700,))
random_idx_test2 = torch.randint(0, 501, (700,))
random_idx_test3 = torch.randint(0, 48, (700,))


random_file = int(sys.argv[1])

random_seed = int(sys.argv[2])

torch.manual_seed(random_seed)
# random.seed(12345)

X, positions = convert_panel(panel_dir + "panel_" + str(random_file))
X = torch.tensor(X).to(device)[:49, start_seq:start_seq + len_seq] # n_ind_adm, input_size
positions = torch.tensor(positions).to(device)[:start_seq + len_seq]

y = convert_split(split_dir + "split_" + str(random_file), positions)
y = torch.tensor(y) # 2 * n_ind_adm, input_size
y = (y[::2] + y[1::2])[:49, start_seq:start_seq + len_seq] # unphase ancestry labels # same shape as X

positions = positions[start_seq:]

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
# print()
# print("Time: ", time.time() - t1)

y_pred /= y_pred.sum(dim=-1, keepdim=True)
# print("Avg certainty: ", y_pred.amax(dim=-1).mean().item())

y_pred = y_pred.argmax(dim=-1)
with open(f"cluster_pred/experiment562/y_pred_" + "_".join(sys.argv[1:]), "w") as f:
    f.write(str(y_pred.cpu().tolist()))
y = y.to(device)
# for i in range(3):
#     print((y_pred == i).sum().item())
#     print((y == i).sum().item())
#     print()


# true values vary left to right
CM = torch.zeros(3, 3)
for i in range(3):
    for j in range(3):
        CM[i, j] = ((y_pred == i) & (y == j)).sum().item()

# print(CM)
# print() 

acc = max((y_pred == y).sum().item(), (2 - y_pred == y).sum().item()) / y.numel()
print(start_seq, sys.argv[3], f"Accuracy: {acc:0.6f}")




if False:
    import matplotlib.pyplot as plt
    plt.imshow((y_pred == y).cpu(), interpolation="nearest", aspect="auto")
    plt.show()