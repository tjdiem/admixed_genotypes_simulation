from globals import *
from processing import *
from kmodels2 import KNet4
import os
from math import log, e
from copy import deepcopy

@torch.no_grad()
def predict_cluster(model, SOI, positions_bp, recombination_map, len_chrom_bp=None, batch_size=batch_size, num_generations=None, admixture_proportion=None, population_size=None):
    torch.manual_seed(2990)
    random.seed(2990)

    loading = False
    save_data = True

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
    num_seeds = 6

    split_chunk_idx = torch.arange(num_chunks + 2).int() * len_seq // (num_chunks + 1) # should technically be based on position not index
    len_seq_chunked = split_chunk_idx[2:] - split_chunk_idx[:-2]

    SOI_chunked = [SOI[:, split_chunk_idx[i]:split_chunk_idx[i+2]] for i in range(num_chunks)]
    positions_chunked = [positions_morgans[split_chunk_idx[i]:split_chunk_idx[i+2]] for i in range(num_chunks)]

    predictions_chunked = [[torch.zeros((num_individuals, len_seq_chunked[i], num_classes)).to(device) for _ in range(num_seeds)] for i in range(num_chunks)]

    refs_chunked = []
    labels_chunked = []

    rand_starts = torch.stack([torch.randperm(num_individuals).long()[:num_seeds] for _ in range(num_chunks)])
    label_mask = (1 - torch.eye(num_individuals)).bool() # is there some way we can make labels a pointer to predictions

    if not loading:
        for chunk in range(num_chunks):
            labels_chunked.append([])
            for seed in range(num_seeds):

                # masks and padding within this for loop can be reused
                # some lines can be written as list comprehension and parallelized
                
                lam = 2 * population_size * (1 - e ** (-num_generations / (2*population_size)))
                lam_a = admixture_proportion * lam
                lam_c = (1 - admixture_proportion) * lam

                positions_morgans_tmp = (positions_chunked[chunk] - positions_chunked[chunk][len_seq_chunked[chunk] // 2]).abs() # should technically be based on position not index
                transition_aa_haploid = lam_c / lam + (lam_a/lam) * torch.exp(-lam * positions_morgans_tmp) # can make this more efficient by multiplying exps  # change in other locations too
                transition_cc_haploid = lam_a / lam + (lam_c/lam) * torch.exp(-lam * positions_morgans_tmp)

                infered_tract0 = 0.5 * (1 - admixture_proportion) * transition_aa_haploid + 0.5 * admixture_proportion * transition_cc_haploid
                infered_tract2 = 0.5 - infered_tract0

                    
                rand_start = rand_starts[chunk][seed]
                predictions_chunked[chunk][seed][:, :, 0] = 0.25 
                predictions_chunked[chunk][seed][:, :, 2] = 0.25
                predictions_chunked[chunk][seed][:, :, 1] = 0.5
                predictions_chunked[chunk][seed][rand_start, :, 0] = infered_tract0
                predictions_chunked[chunk][seed][rand_start, :, 2] = infered_tract2

                padding = torch.full((num_individuals, input_size // 2, num_classes), 0).to(device)
                predictions_chunked[chunk][seed] = torch.cat((padding, predictions_chunked[chunk][seed], padding), dim=1) # (num_individuals, len_seq + input_size - 1, num_classes)
                labels_chunked[-1].append(predictions_chunked[chunk][seed].unsqueeze(0).expand(num_individuals,-1,-1,-1)[label_mask].reshape(num_individuals, num_individuals - 1, len_seq_chunked[chunk] + input_size - 1, num_classes))

            padding = torch.full((num_individuals, input_size // 2), -1).to(device)
            SOI_chunked[chunk] = torch.cat((padding, SOI_chunked[chunk], padding), dim=1) # (num_individuals, len_seq + input_size - 1)

            padding = torch.full((input_size // 2,), float("inf")).to(device)
            positions_chunked[chunk] = torch.cat((padding, positions_chunked[chunk], padding), dim=0)

            refs_chunked.append(SOI_chunked[chunk].unsqueeze(0).expand(num_individuals,-1,-1)[label_mask].reshape(num_individuals, num_individuals -1 , len_seq_chunked[chunk] + input_size - 1))

        for chunk in range(num_chunks):

            for seed in range(num_seeds):

                for i in range(0, num_individuals * len_seq_chunked[chunk], batch_size):

                    probabilities = 1 - predictions_chunked[chunk][seed][:, input_size // 2 : - (input_size // 2)].max(dim=-1)[0].cpu()
                    if i == 0: ###
                        probabilities[rand_starts[chunk][seed]] = 0
                    ind1 = torch.multinomial(probabilities.sum(dim=-1), batch_size, replacement=False)
                    ind2 = torch.multinomial(probabilities[ind1], 1).squeeze(-1)
                    
                    ind3, ind4 = ind1.clone(), ind2.clone() #############


                    ind1 = ind1.unsqueeze(-1)  #faster way to index this?
                    ind2 = ind2.unsqueeze(-1) + torch.arange(input_size).long()
                    SOI_batch = SOI_chunked[chunk][ind1, ind2]
                    refs_batch = refs_chunked[chunk][ind1, :, ind2].transpose(1,2)
                    labels_batch = labels_chunked[chunk][seed][ind1, :, ind2].transpose(1,2)
                    positions_batch = positions_chunked[chunk].unsqueeze(0).expand(batch_size, -1)[torch.arange(batch_size).long().unsqueeze(-1), ind2]

                    params_batch = torch.full((batch_size, 6), num_generations).to(device) # fix this

                    if save_data:

                        if random.random() < batch_size * 0.02:

                            batch_ind = random.randint(0, batch_size - 1)

                            file_id = 0
                            while True:
                                file_name = "cluster_training_data/SOI_file_batch_" + str(file_id)
                                if not os.path.exists(file_name):
                                    break
                                file_id += 1

                            torch.save(SOI_batch[batch_ind], "cluster_training_data/SOI_file_batch_" + str(file_id))
                            torch.save(refs_batch[batch_ind], "cluster_training_data/refs_file_batch_" + str(file_id))
                            torch.save(labels_batch[batch_ind], "cluster_training_data/labels_file_batch_" + str(file_id))
                            torch.save(positions_batch[batch_ind], "cluster_training_data/positions_file_batch_" + str(file_id))
                            torch.save(params_batch[batch_ind], "cluster_training_data/params_file_batch_" + str(file_id))

                            y_chunk = y[:, split_chunk_idx[chunk]:split_chunk_idx[chunk+2]]

                            switched = predictions_chunked[chunk][seed][:, input_size // 2: -(input_size // 2)][y_chunk == 0][:, [0,2]].sum(dim=0) + predictions_chunked[chunk][seed][:, input_size // 2: -(input_size // 2)][y_chunk == 2][:, [2,0]].sum(dim=0)


                            if switched[-1] > switched[0]:
                                out_true = 2 - y_chunk[ind3[batch_ind], ind4[batch_ind]]

                            else:
                                out_true = y_chunk[ind3[batch_ind], ind4[batch_ind]]

                            torch.save(out_true, "cluster_training_data/out_file_batch_" + str(file_id))

                    out = model(SOI_batch, refs_batch, labels_batch, positions_batch, params_batch)
                    out = F.softmax(out, dim=-1).double() # batch, num_classes

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
                    
                    # we should have separate padded and unpadded tensors.  They can be pointers to each other
                    predictions_chunked[chunk][seed][ind3] = predictions_chunked[chunk][seed][ind3] * (1 - tmp) + out_smoothed * tmp
                    predictions_chunked[chunk][seed][:, :input_size // 2] = 0
                    predictions_chunked[chunk][seed][:, -(input_size // 2):] = 0

                    assert ((1 - predictions_chunked[chunk][seed][:, input_size // 2: - (input_size // 2)].sum(dim=-1)).abs() < 1e-4).all()

                    labels_chunked[chunk][seed] = predictions_chunked[chunk][seed].unsqueeze(0).expand(num_individuals,-1,-1,-1)[label_mask].reshape(num_individuals, num_individuals - 1, len_seq_chunked[chunk] + input_size - 1, num_classes)

                if not save_data:
                    torch.save(predictions_chunked[chunk][seed], f"cluster_pred/predict5/{random_file}_chunk{chunk}_start{seed}")


    else: # if loading:

        if False:
            predictions_chunked = []
            for chunk in range(num_chunks):
                predictions_chunked.append([])
                for seed in range(num_seeds):
                    predictions_chunked[-1].append(torch.load(f"cluster_pred/predict5/{random_file}_chunk{chunk}_start{seed}"))

                    with open(f"cluster_pred/experiment562/y_pred_{random_file}_202_{seed}_{chunk * 250}", "r") as f:
                        predictions_chunked[-1].append(torch.tensor(eval(f.read())))

                    print(predictions_chunked[-1][-1].shape)


        else:
            ### if from predict_cluster2_exp
            predictions_chunked = []
            for chunk in range(num_chunks):
                predictions_chunked.append([])
                for seed in range(num_seeds):

                    with open(f"cluster_pred/experiment562/y_pred_{random_file}_202_{seed}_{chunk * 250}", "r") as f:
                        tmp = torch.tensor(eval(f.read())).to(device)
                    
                    tmp = F.one_hot(tmp, num_classes=num_classes)
                    padding = torch.zeros((49, 250, 3)).to(device)
                    tmp = torch.cat((padding, tmp, padding), dim=1)
                    predictions_chunked[-1].append(tmp)

            ###
    

        predictions_strict_chunked = []
        for chunk in range(num_chunks):
            predictions_strict_chunked.append([])
            for seed in range(num_seeds):
                predictions_strict_chunked[-1].append(predictions_chunked[chunk][seed][:, input_size // 2: -(input_size // 2)].argmax(dim=-1))

    if not save_data:
        accuracies = []
        for chunk in range(num_chunks):
            accuracies.append([])
            chunk_start = chunk * 250
            for seed in range(num_seeds):
                y_pred = predictions_strict_chunked[chunk][seed]
                y_true = y[:, chunk_start: chunk_start + 500].to(device)

                acc = max((y_pred == y_true).float().mean().item(), (y_pred == 2 - y_true).float().mean().item())
                print(chunk, seed, acc)
                accuracies[-1].append(acc)
            
        transitions = torch.zeros((num_chunks - 1, num_seeds, num_seeds))
        is_flipped = torch.zeros((num_chunks - 1, num_seeds, num_seeds)).int()
        # this loop could be done a lot more efficiently
        for chunk in range(num_chunks - 1):
            for num1 in range(num_seeds):   
                for num2 in range(num_seeds):
                    left_pred = predictions_strict_chunked[chunk]
                    right_pred = predictions_strict_chunked[chunk + 1]               
                    similarity, similarity_flipped = (left_pred[num1][:, 250:] == right_pred[num2][:, :250]).float().mean(), (left_pred[num1][:, 250:] == 2 - right_pred[num2][:, :250]).float().mean()
                    
                    # left_pred = predictions_chunked[chunk]
                    # right_pred = predictions_chunked[chunk + 1]

                    # similarity = (left_pred[num1][:, 505:745:20] * right_pred[num2][:, 255:495:20]).sum(dim=-1).log().sum()
                    # similarity_flipped = (left_pred[num1][:, 505:745:20] * right_pred[num2][...,[2,1,0]][:, 255:495:20]).sum(dim=-1).log().sum()
                    
                    
                    if similarity > similarity_flipped:
                        transitions[chunk, num1, num2] = similarity
                        is_flipped[chunk, num1, num2] = 0
                    else:
                        transitions[chunk, num1, num2] = similarity_flipped
                        is_flipped[chunk, num1, num2] = 1

        for i in range(num_chunks - 1):
            printing = torch.zeros((num_seeds + 1, num_seeds + 1))
            printing[1:, 1:] = transitions[i]
            printing[0, 1:] = torch.tensor(accuracies[i + 1])
            printing[1:, 0] = torch.tensor(accuracies[i])

            print(printing)

        # import matplotlib.pyplot as plt
        # fig, axes = plt.subplots(4, 4)
        # axes = axes.flatten()
        # for image, ax in zip(predictions_strict_chunked[-2] + predictions_strict_chunked[-1], axes):
        #     im = ax.imshow(image.cpu(), interpolation="nearest", aspect="auto")

        # plt.subplots_adjust(wspace=0.2, hspace=0.6)
        # fig.colorbar(im, ax=axes, orientation="vertical",  fraction = 0.02, pad=0.04)
        # plt.show()
        # exit()


        # write how to combine predictions
        # can we tell if accuracy is bad from transition matrices?
        # if not:
            # calculate similarity in different way?
            # can we write likelihood function for each prediction alone
        # does changing predicted index probability change anything?
            # use reinforcement learning to choose prediction order?
        # does training on probabilistic data chcange anything? - probably less likely to change anything

        best_val = transitions[-1].amax(dim=-1)
        args = [transitions[-1].argmax(dim=-1)]
        for i in range(len(transitions) - 2, -1, -1):
            transitions[i] += best_val
            best_val = transitions[i].amax(dim=-1)
            args.append(transitions[i].argmax(dim=-1))

        args = args[::-1]

        arg = best_val.argmax()
        best_val = best_val.amax()
        best_args = [arg.item()]
        for i in range(len(args)):
            arg = args[i][arg].item()
            best_args.append(arg)

        is_flipped = torch.tensor([0] + [is_flipped[chunk][best_args[chunk], best_args[chunk+1]].item() for chunk in range(num_chunks - 1)])
        is_flipped = is_flipped.cumsum(0) % 2

        print(best_val)
        print(best_args)

        print(is_flipped)

        # for arg, acc in zip(best_args, accuracies):
        #     print(acc[arg])

        print(best_args)
        predictions = torch.zeros((num_individuals, len_seq, num_classes)).to(device)
        for chunk in range(num_chunks):
            predicted_chunk = predictions_chunked[chunk][best_args[chunk]][:, input_size // 2: -(input_size // 2)]
            if is_flipped[chunk]:
                predicted_chunk = predicted_chunk[...,[2,1,0]]
            predictions[:, chunk * 250: chunk * 250 + 500] += predicted_chunk

        predictions[...,1] *= 1.001
        predictions[...,2] *= 1.002
        return predictions
    
        if True:
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


        return predictions

                                             
len_seq = 2750

random_file = int(sys.argv[1])

torch.manual_seed(409)
# random.seed(12345)

X, positions = convert_panel(panel_dir + "panel_" + str(random_file))
X = torch.tensor(X).to(device)[:49, :len_seq] # n_ind_adm, input_size
positions = torch.tensor(positions).to(device)[:len_seq]

if False:
    y = convert_split(split_dir + "split_" + str(random_file), positions)
    y = torch.tensor(y) # 2 * n_ind_adm, input_size
    y = (y[::2] + y[1::2])[:49, :len_seq] # unphase ancestry labels # same shape as X
else:
    y = torch.cat([torch.load(f"saved_inputs/y_file{random_file}_chunk{chunk}.pt") for chunk in range(0, len_seq, 1000)], dim=-1)[:49, :len_seq].to(device)

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

y_pred /= y_pred.sum(dim=-1, keepdim=True)
print("Avg certainty: ", y_pred.amax(dim=-1).mean().item())

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

# y_pred = y_pred[:, 250: -250]
# y = y[:, 250: -250]
# acc = max((y_pred == y).sum().item(), (2 - y_pred == y).sum().item()) / y.numel()
# print(f"Accuracy: {acc:0.6f}")


# import matplotlib.pyplot as plt
# plt.imshow((y_pred == y).sum(dim=0, keepdim=True).cpu(), interpolation="nearest", aspect="auto")
# plt.colorbar()
# plt.show()