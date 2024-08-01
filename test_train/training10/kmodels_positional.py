from globals import *
from math import e, sqrt, log
from scipy.optimize import linear_sum_assignment
import numpy as np
import TNet

OHE_values = torch.arange(3**4).reshape(3, 3, 3, 3) # SOI value, ref value, class we are in, labeled class of ref
OHE_values[:, :, 2] = torch.flip(OHE_values[:, :, 0], dims=(2,))
OHE_values[2] = torch.flip(OHE_values[0], dims=(0,))
unique_elements, inverse_indices = torch.unique(OHE_values, return_inverse=True)
OHE_values = torch.arange(len(unique_elements))[inverse_indices]
OHE_values = OHE_values.flatten()

assert n_embd == OHE_values.max() + 1

Transition = torch.zeros((num_classes**4, n_embd))
Transition[torch.arange(num_classes**4).long(), OHE_values] = 1 #dype = long


# how to index:
# labeled class + class we are in * 3 + ref_value * 9 + SOI value * 27
#    var1              var2                  var3           var4
    
# if var3_1 == var3_2, var4_1 == var4_2, var2_1 == 2 - var2_2 != 1, var1_1 == 2 - var1_2
    # then values should be same
# else they should be different

class KNet_positional(nn.Module):
    def __init__(self):
        super().__init__()

        hidden0 = n_embd_model

        hidden1 = 1000
        hidden2 = 100

        hidden3 = 50

        self.linear0 = nn.Linear(n_embd_model, hidden0)

        self.linear1 = nn.Linear(input_size_positional * hidden0, hidden1)
        self.linear2 = nn.Linear(hidden1, hidden2) 
        self.linear3 = nn.Linear(hidden2, 1)

        self.linear4 = nn.Linear((num_classes * n_ind_pan_model), hidden3)
        self.linear5 = nn.Linear(hidden3, 1)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    @torch.no_grad()
    def predict_cluster(self, SOI, batch_size=batch_size):

        num_individuals, len_seq = SOI.shape

        padding = torch.full((num_individuals, input_size_positional // 2), -1).to(device)
        SOI = torch.cat((padding, SOI, padding), dim=1) # (num_individuals, len_seq + input_size_positional - 1)
        
        # fill with 0.25, 0.5, 0.25 # fill with ancestry proporotion
        predictions = torch.full((num_individuals, len_seq, num_classes), 1/num_classes).to(device) # (num_individuals, len_seq, num_classes)

        ######
        # hard coded for 2 ancestries.  # make sure formula is correct
        # should be based on positions
        ### ChatGPT: The CDF, which gives the probability that a tract length is less than or equal to somve value L, is: F(L) = 1 - e^(-NL)
        num_generations = 20 
        predictions[0, :, :2] = torch.exp(torch.arange(len_seq) * (-num_generations/100)).unsqueeze(-1).repeat(1,2) * (1/6) + (1/3)
        predictions[0, :, 2] = 1 - predictions[0, :, 0] - predictions[0, :, 1]

        padding = torch.full((num_individuals, input_size_positional // 2, num_classes), 0).to(device)
        predictions = torch.cat((padding, predictions, padding), dim=1) # (num_individuals, len_seq + input_size_positional - 1, num_classes)

        mask = (1 - torch.eye(num_individuals)).bool() # is there some way we can make labels a pointer to predictions
        refs = SOI.unsqueeze(0).expand(num_individuals,-1,-1)[mask].reshape(num_individuals, num_individuals -1 , len_seq + input_size_positional - 1)
        labels = predictions.unsqueeze(0).expand(num_individuals,-1,-1,-1)[mask].reshape(num_individuals, num_individuals - 1, len_seq + input_size_positional - 1, num_classes)

        print("shapes")
        print(SOI.shape)
        print(refs.shape)
        print(predictions.shape)
        print(labels.shape)

        refs2 = torch.zeros(49, 48, 950).to(device)
        for i in range(49):
            refs2[i] = torch.cat((SOI[:i], SOI[i+1:]), dim=0)
        print(torch.equal(refs, refs2))

        ####
        # while True:
        #     for istart in range(1, num_individuals, max_batch_size):
        #         iend = min(istart + max_batch_size, num_individuals - 1)
        #         out = self(SOI[istart:iend, :input_size_positional], refs[istart:iend, :, :input_size_positional], labels[istart:iend, :, :input_size_positional])
        #         predictions[istart:iend, input_size_positional // 2] = F.softmax(out, dim=-1)
        #         # right now the forward method chooses 48 random ref panels

            
            
        #     # unnecessary if we can make labels a pointer to predictions
        #     # indent/unindent this
        #     labels = predictions.unsqueeze(0).expand(num_individuals,-1,-1,-1)[mask].reshape(num_individuals, num_individuals - 1, len_seq + input_size_positional - 1, num_classes)
        ####

        """
        improvement:

        different way to chooose next predicted index:
            structured order
            random but based on last time index was predicted
            based on how much it changed last time
            reinforcement learning specifically for this

        include number generations and admixture proportion prediction in the bootstrapping method
            use num generations to smooth prediction
            use admixture proportion to determine class frequencies
            iteratively update these estimates

        include smoothing

        use posterior probabilities in combination with new predictions
            posterior probabilities can be previous predictions and/or class proportions

        use positions in model
        
        """
            
        when_predicted = torch.full((num_individuals, len_seq), float("inf"))
        s = 0
        for i in range(0, num_individuals * len_seq, batch_size):
            ind1 = torch.randint(0, num_individuals, (batch_size,))
            ind2 = torch.randint(0, len_seq, (batch_size,))

            when_predicted += 1
            when_predicted[ind1, ind2] = 0

            ind1 = ind1.unsqueeze(-1)
            ind2 = ind2.unsqueeze(-1) + torch.arange(input_size_positional).long()
            out = self(SOI[ind1, ind2], refs[ind1, :, ind2].transpose(1,2), labels[ind1, :, ind2].transpose(1,2))
            out = F.softmax(out, dim=-1)
            # torch.stack indices instead?  

            s += (predictions[ind1[:,0], ind2[:,0] + input_size_positional // 2] - out).abs().sum().item()
            predictions[ind1[:,0], ind2[:,0] + input_size_positional // 2] = out

            labels = predictions.unsqueeze(0).expand(num_individuals,-1,-1,-1)[mask].reshape(num_individuals, num_individuals - 1, len_seq + input_size_positional - 1, num_classes)

            # if i % 100 == 0:
            #     print(i/ (num_individuals * len_seq * 6))
            #     print(s/100)
            #     print(when_predicted)
            #     s = 0


        return predictions[:, input_size_positional // 2: -(input_size_positional // 2)]
    
    @torch.no_grad()
    def predict_cluster2(self, SOI, positions, batch_size=batch_size, num_generations=None):
        
        if num_generations is None:
            num_generations = 20 #### Change this default value
            infer_num_generations = True
        else:
            infer_num_generations = False

        num_individuals, len_seq = SOI.shape

        padding = torch.full((num_individuals, input_size_positional // 2), -1).to(device)
        SOI = torch.cat((padding, SOI, padding), dim=1) # (num_individuals, len_seq + input_size_positional - 1)
        
        # fill with 0.25, 0.5, 0.25 # fill with ancestry proporotion
        predictions = torch.full((num_individuals, len_seq, num_classes), 1/num_classes).to(device) # (num_individuals, len_seq, num_classes)

        predictions[...,[0,2]] = 0.25
        predictions[...,1] = 0.5
        predictions[0, :, 0] = torch.exp(-num_generations * positions) * (1/4) + (1/4)
        # predictions[0, :, 0] = torch.exp(torch.arange(len_seq) * (-num_generations/100)) * (1/4) + (1/4)
        predictions[0, :, 1] = 0.5
        predictions[0, :, 2] = 1 - predictions[0, :, 0] - predictions[0, :, 1]

        padding = torch.full((num_individuals, input_size_positional // 2, num_classes), 0).to(device)
        predictions = torch.cat((padding, predictions, padding), dim=1) # (num_individuals, len_seq + input_size_positional - 1, num_classes)

        padding = torch.full((input_size_positional // 2,), float("inf")).to(device)
        positions = torch.cat((padding, positions / num_bp, padding), dim=0)

        mask = (1 - torch.eye(num_individuals)).bool() # is there some way we can make labels a pointer to predictions
        refs = SOI.unsqueeze(0).expand(num_individuals,-1,-1)[mask].reshape(num_individuals, num_individuals -1 , len_seq + input_size_positional - 1)
        labels = predictions.unsqueeze(0).expand(num_individuals,-1,-1,-1)[mask].reshape(num_individuals, num_individuals - 1, len_seq + input_size_positional - 1, num_classes)

        print("shapes")
        print(SOI.shape)
        print(refs.shape)
        print(predictions.shape)
        print(labels.shape)

        refs2 = torch.zeros(49, 48, 950).to(device)
        for i in range(49):
            refs2[i] = torch.cat((SOI[:i], SOI[i+1:]), dim=0)
        print(torch.equal(refs, refs2))

        strict_predictions = predictions.clone()
            
        when_predicted = torch.full((num_individuals, len_seq), 1.0)
        has_predicted = torch.zeros((num_individuals, len_seq))

        for i in range(0, num_individuals * len_seq, batch_size):
            probabilities = (when_predicted / when_predicted.sum()).flatten()
            ind = torch.multinomial(probabilities, batch_size, replacement=False)
            
            ind1 = ind // len_seq
            ind2 = ind % len_seq

            ind3, ind4 = ind1.clone(), ind2.clone() #############

            # ind1 = torch.randint(0, num_individuals, (batch_size,))
            # ind2 = torch.randint(0, len_seq, (batch_size,))

            has_predicted[ind1, ind2] = 1

            ind1 = ind1.unsqueeze(-1)
            ind2 = ind2.unsqueeze(-1) + torch.arange(input_size_positional).long()
            SOI_batch = SOI[ind1, ind2]
            refs_batch = refs[ind1, :, ind2].transpose(1,2)
            labels_batch = labels[ind1, :, ind2].transpose(1,2)
            positions_batch = positions.unsqueeze(0).expand(batch_size, -1)[torch.arange(batch_size).long().unsqueeze(-1), ind2]
            # torch.stack indices instead?  
            params_batch = torch.full((batch_size, 6), num_generations).to(device)

            out = self(SOI_batch, refs_batch, labels_batch, positions_batch, params_batch)
            out = F.softmax(out, dim=-1) # batch, num_classes

            # conider multiplying exp distribution by alpha?
            len_exp_distribution = 49 ## this should be chosen based on num generations and threshold accuracy
            positions_batch = (positions_batch[:, input_size_positional // 2 - len_exp_distribution // 2: input_size_positional // 2 + len_exp_distribution // 2 + 1] - positions_batch[:, input_size_positional // 2].unsqueeze(-1)).abs()
            # The below line is not exactly right.  Each predicted class could transition to another class with different probabilities
            exp_distribution = torch.exp(-2 * num_generations * (positions_batch)) # batch, len_exp_distribution

            out_smoothed = out.unsqueeze(1) * exp_distribution.unsqueeze(-1) # batch, len_exp_distribution, num_classes
            predictions_idx = torch.stack([predictions[ind1[j,0], ind2[j,0] + input_size_positional // 2 - len_exp_distribution // 2: ind2[j,0] + input_size_positional // 2 + len_exp_distribution // 2 + 1] for j in range(batch_size)])
            predictions_smoothed = predictions_idx * (1 - exp_distribution.unsqueeze(-1))

            for j in range(batch_size):
                predictions[ind1[j,0], ind2[j,0] + input_size_positional // 2 - len_exp_distribution // 2: ind2[j,0] + input_size_positional // 2 + len_exp_distribution // 2 + 1] = out_smoothed[j] + predictions_smoothed[j]

            labels = predictions.unsqueeze(0).expand(num_individuals,-1,-1,-1)[mask].reshape(num_individuals, num_individuals - 1, len_seq + input_size_positional - 1, num_classes)

            #####
            strict_predictions[ind3, ind4] = out

            if infer_num_generations and i % (batch_size * 100) == 0 and i > 0:

                num_tracts = (predictions[:, input_size_positional // 2: -(input_size_positional // 2) - 1].argmax(dim=-1) != predictions[:, input_size_positional // 2 + 1: -(input_size_positional // 2)].argmax(dim=-1)).sum().item() + num_individuals
                print(num_tracts)
                avg_len_transition = num_individuals * (5.8413e-02 - 5.1200e-06) / num_tracts
                print(avg_len_transition)

                num_generations = 1 / (4 * avg_len_transition)
                print(num_generations)
                print()

                # total_length = 0
                # total_n = 0
                # for j in range(num_individuals):
                #     positions_valid = positions[input_size_positional // 2 : - (input_size_positional // 2)][has_predicted[j].bool()]
                #     predictions_valid = strict_predictions[j, input_size_positional // 2 : - (input_size_positional // 2)][has_predicted[j].bool()].argmax(dim=-1)
                #     # print(positions_valid)
                #     # print(predictions_valid)
                #     transitions = predictions_valid[:-1] != predictions_valid[1:]
                #     transition_positions = (positions_valid[:-1][transitions] + positions_valid[1:][transitions]) / 2
                #     # this doesn't include first and last ancestry tracts!  Should I include this?
                #     tract_lengths = transition_positions[1:] - transition_positions[:-1]
                #     total_length += tract_lengths.sum().item()
                #     total_n += tract_lengths.shape[0]

                # print()
                # print(total_length)
                # print(total_n)
                # num_generations = total_n / (total_length * 4)
                # print(num_generations)



                continue
                from scipy.optimize import curve_fit

                def exp_model(dist, a, b, num_generations):
                    return (1 - dist) ** num_generations * a + b

                print("infer num generations")
                # max dist should be based on num generations (current prediction) and threshold
                max_dist_valid = 1e-3
                all_predictions = predictions[:, input_size_positional // 2 : - (input_size_positional // 2)][has_predicted.bool()]
                probs_cov = torch.cov(all_predictions.t())
                print(probs_cov)
                for j in range(num_individuals):
                    positions_predicted = positions[input_size_positional // 2 : - (input_size_positional // 2)][has_predicted[j].bool()]
                    distances = positions_predicted.unsqueeze(0) - positions_predicted.unsqueeze(1)
                    valid_pairs = (distances > 0) & (distances < max_dist_valid)
                    valid_pairs = valid_pairs.nonzero(as_tuple = True)

                    # valid_predictions = predictions[j, input_size_positional // 2 : - (input_size_positional // 2)][has_predicted[j].bool()]
                    #####
                    valid_predictions = strict_predictions[j, input_size_positional // 2 : - (input_size_positional // 2)][has_predicted[j].bool()]
                    #we have to account for correlated probabilities
                    # we have to update this iteratively. weighted average between new estimate and old estimate where new estimate only includes newer predictions
                    probs_same = valid_predictions[valid_pairs[0]] * valid_predictions[valid_pairs[1]]
                    valid_distances = distances[valid_pairs]

                    valid_pairs_all = distances.abs() < max_dist_valid
                    print(valid_pairs_all)

                    print(valid_pairs)
                    
                    y = valid_predictions[:5]
                    x = positions_predicted[:5]
                    print(y)

                    y = y[:, 0].cpu().numpy()
                    x = (x - x[0]).cpu().numpy()

                    print(x)
                    print(y)

                    popt, pcov = curve_fit(exp_model, x, y)
                    print(popt, pcov)


                    exit()

                    
                # positions_predicted = positions.unsqueeze(0).expand(num_individuals, -1)[:, input_size_positional // 2: -(input_size_positional // 2)][has_predicted.bool()]

                # print(positions_predicted.shape)
                

        return predictions[:, input_size_positional // 2: -(input_size_positional // 2)]

    @torch.no_grad()
    def predict_cluster3(self, SOI, positions_bp, recombination_map, len_chrom_bp=None, batch_size=batch_size, num_generations=None, admixture_proportion=None, population_size=None):
        
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

        padding = torch.full((num_individuals, input_size_positional // 2), -1).to(device)
        SOI = torch.cat((padding, SOI, padding), dim=1) # (num_individuals, len_seq + input_size_positional - 1)
        
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

        # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        # 0.5 0.5 instead of 1 - ap, ap ??!!!
        # predictions[0, :, 0] = 0.5 * (1 - admixture_proportion) * transition_aa_haploid + 0.5 * admixture_proportion * transition_cc_haploid
        # predictions[0, :, 2] = 0.5 - predictions[0, :, 0]

        padding = torch.full((num_individuals, input_size_positional // 2, num_classes), 0).to(device)
        predictions = torch.cat((padding, predictions, padding), dim=1) # (num_individuals, len_seq + input_size_positional - 1, num_classes)

        padding = torch.full((input_size_positional // 2,), float("inf")).to(device)
        positions = torch.cat((padding, positions_morgans, padding), dim=0)

        mask = (1 - torch.eye(num_individuals)).bool() # is there some way we can make labels a pointer to predictions
        refs = SOI.unsqueeze(0).expand(num_individuals,-1,-1)[mask].reshape(num_individuals, num_individuals -1 , len_seq + input_size_positional - 1)
        labels = predictions.unsqueeze(0).expand(num_individuals,-1,-1,-1)[mask].reshape(num_individuals, num_individuals - 1, len_seq + input_size_positional - 1, num_classes)

        print("shapes")
        print(SOI.shape)
        print(refs.shape)
        print(predictions.shape)
        print(labels.shape)

        refs2 = torch.zeros(num_individuals, num_individuals - 1, len_seq + input_size_positional - 1).to(device)
        for i in range(num_individuals):
            refs2[i] = torch.cat((SOI[:i], SOI[i+1:]), dim=0)
        print(torch.equal(refs, refs2))

        strict_predictions = predictions.clone()
            
        when_predicted = torch.full((num_individuals, len_seq), 1.0)
        has_predicted = torch.zeros((num_individuals, len_seq))

        for i in range(0, num_individuals * len_seq, batch_size):

            # probabilities = (when_predicted / when_predicted.sum()).flatten()
            # ind = torch.multinomial(probabilities, batch_size, replacement=False)
            # ind1 = ind // len_seq
            # ind2 = ind % len_seq

            # probabilities = (when_predicted / when_predicted.sum())
            if i == 0:
                probabilities[0] = 0

            probabilities = 1 - predictions[:, input_size_positional // 2 : - (input_size_positional // 2)].max(dim=-1)[0].cpu()
            ind1 = torch.multinomial(probabilities.sum(dim=-1), batch_size, replacement=False)
            ind2 = torch.multinomial(probabilities[ind1], 1).squeeze(-1)
            
            ind3, ind4 = ind1.clone(), ind2.clone() #############

            # ind1 = torch.randint(0, num_individuals, (batch_size,))
            # ind2 = torch.randint(0, len_seq, (batch_size,))

            has_predicted[ind1, ind2] = 1

            ind1 = ind1.unsqueeze(-1)
            ind2 = ind2.unsqueeze(-1) + torch.arange(input_size_positional).long()
            SOI_batch = SOI[ind1, ind2]
            refs_batch = refs[ind1, :, ind2].transpose(1,2)
            labels_batch = labels[ind1, :, ind2].transpose(1,2)
            positions_batch = positions.unsqueeze(0).expand(batch_size, -1)[torch.arange(batch_size).long().unsqueeze(-1), ind2]
            # torch.stack indices instead?  
            params_batch = torch.full((batch_size, 6), num_generations).to(device)

            out = self(SOI_batch, refs_batch, labels_batch, positions_batch, params_batch)
            out = F.softmax(out, dim=-1).double() # batch, num_classes

            ###########9999
            positions_diff = (positions - positions_batch[:, input_size_positional // 2].unsqueeze(-1)).abs().to(device) # batch, len_seq + input_size_positional

            transition_aa_haploid = lam_c / lam + (lam_a/lam) * torch.exp(-lam * positions_diff) # batch, len_seq + input_size_positional - 1
            transition_cc_haploid = lam_a / lam + (lam_c/lam) * torch.exp(-lam * positions_diff)
            transition_ac_haploid = 1 - transition_aa_haploid
            transition_ca_haploid = 1 - transition_cc_haploid
            
            transitions = torch.zeros((batch_size, len_seq + input_size_positional - 1, num_classes, num_classes)).double().to(device)
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
            tmp = torch.exp(-2 * num_generations * positions_diff * 10).unsqueeze(-1) #hardcoded for now
            
            predictions[ind3] = predictions[ind3] * (1 - tmp) + out_smoothed * tmp
            predictions[:, :input_size_positional // 2] = 0
            predictions[:, -(input_size_positional // 2):] = 0

            assert ((1 - predictions[:, input_size_positional // 2: - (input_size_positional // 2)].sum(dim=-1)).abs() < 1e-4).all()
            ###########9999

            ###########9999
            # conider multiplying exp distribution by alpha?
            # out = out.float()
            # len_exp_distribution = 49 ## this should be chosen based on num generations and threshold accuracy
            # positions_batch = (positions_batch[:, input_size_positional // 2 - len_exp_distribution // 2: input_size_positional // 2 + len_exp_distribution // 2 + 1] - positions_batch[:, input_size_positional // 2].unsqueeze(-1)).abs()
            # # The below line is not exactly right.  Each predicted class could transition to another class with different probabilities
            # exp_distribution = torch.exp(-2 * num_generations * (positions_batch)) # batch, len_exp_distribution

            # out_smoothed = out.unsqueeze(1) * exp_distribution.unsqueeze(-1) # batch, len_exp_distribution, num_classes
            # predictions_idx = torch.stack([predictions[ind1[j,0], ind2[j,0] + input_size_positional // 2 - len_exp_distribution // 2: ind2[j,0] + input_size_positional // 2 + len_exp_distribution // 2 + 1] for j in range(batch_size)])
            # predictions_smoothed = predictions_idx * (1 - exp_distribution.unsqueeze(-1))

            # for j in range(batch_size):
            #     predictions[ind1[j,0], ind2[j,0] + input_size_positional // 2 - len_exp_distribution // 2: ind2[j,0] + input_size_positional // 2 + len_exp_distribution // 2 + 1] = out_smoothed[j] + predictions_smoothed[j]
            ###########9999

            labels = predictions.unsqueeze(0).expand(num_individuals,-1,-1,-1)[mask].reshape(num_individuals, num_individuals - 1, len_seq + input_size_positional - 1, num_classes)

            #####
            strict_predictions[ind3, ind4] = out

            if infer_num_generations and i % (batch_size * 500) == 0 and i > 0:

                predictions_argmax = predictions[:, input_size_positional // 2: -(input_size_positional // 2)].argmax(dim=-1)

                positions_diff = positions[input_size_positional // 2: -(input_size_positional // 2)]
                positions_diff_start = torch.tensor([(positions_diff[1] - positions_diff[0]) / 2 - 0]).to(device) # this assumes start morgans is 0
                positions_diff_end = torch.tensor([len_chrom_morgan - (positions_diff[-1] + positions_diff[-2]) / 2]).to(device)
                positions_diff = (positions_diff[2:] - positions_diff[:-2]) / 2
                positions_diff = torch.cat((positions_diff_start, positions_diff, positions_diff_end))
                positions_diff = positions_diff.unsqueeze(0).expand(num_individuals, -1)

                proportion_0 = positions_diff[predictions_argmax == 0].sum().item() / (len_chrom_morgan * num_individuals)
                proportion_1 = positions_diff[predictions_argmax == 1].sum().item() / (len_chrom_morgan * num_individuals) 
                proportion_2 = positions_diff[predictions_argmax == 2].sum().item() / (len_chrom_morgan * num_individuals)

                # admixture_proportion_prediction0 = sqrt(proportion_0)
                # admixture_proportion_prediction2 = 1 - sqrt(proportion_2)
                
                # if proportion_1 > 0.5:
                #     admixture_proportion_prediction1 = 0.5
                # elif admixture_proportion_prediction0 + admixture_proportion_prediction2 > 1:
                #     admixture_proportion_prediction1 = (1 + sqrt(1 - 2*proportion_1)) / 2
                # else:
                #     admixture_proportion_prediction1 = (1 - sqrt(1 - 2*proportion_1)) / 2

                # admixture_proportion = (admixture_proportion_prediction0 + admixture_proportion_prediction1 + admixture_proportion_prediction2) / 3

                admixture_proportion = proportion_0 + proportion_1 * 0.5 + proportion_2 * 0

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

        return predictions[:, input_size_positional // 2: -(input_size_positional // 2)]
    
    @torch.no_grad()
    def predict_cluster4(self, SOI, positions_bp, recombination_map, batch_size=batch_size, num_generations=None, admixture_proportion=None, population_size=None):
        
        if num_generations is None:
            num_generations = 20 #### Change this default value
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

        padding = torch.full((num_individuals, input_size_positional // 2), -1).to(device)
        SOI = torch.cat((padding, SOI, padding), dim=1) # (num_individuals, len_seq + input_size_positional - 1)
        
        # fill with 0.25, 0.5, 0.25 # fill with ancestry proporotion
        predictions = torch.full((num_individuals, len_seq, num_classes), 1/num_classes).to(device) # (num_individuals, len_seq, num_classes)

        positions_morgans = recombination_map(positions_bp)

        lam = 2 * population_size * (1 - e ** (-num_generations / (2*population_size)))
        lam_a = admixture_proportion * lam
        lam_c = (1 - admixture_proportion) * lam

        transition_aa_haploid = lam_c / lam + (lam_a/lam) * torch.exp(-lam * positions_morgans) # can make this more efficient by multiplying exps  # change in other locations too
        transition_cc_haploid = lam_a / lam + (lam_c/lam) * torch.exp(-lam * positions_morgans)

        predictions[...,[0,2]] = (admixture_proportion ** 2 + (1 - admixture_proportion) ** 2) / 2
        predictions[...,1] = admixture_proportion * (1 - admixture_proportion) * 2

        # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        # 0.5 0.5 instead of 1 - ap, ap ??!!!
        predictions[0, :, 0] = 0.5 * (1 - admixture_proportion) * transition_aa_haploid + 0.5 * admixture_proportion * transition_cc_haploid
        predictions[0, :, 2] = 0.5 - predictions[0, :, 0]

        padding = torch.full((num_individuals, input_size_positional // 2, num_classes), 0).to(device)
        predictions = torch.cat((padding, predictions, padding), dim=1) # (num_individuals, len_seq + input_size_positional - 1, num_classes)

        padding = torch.full((input_size_positional // 2,), float("inf")).to(device)
        positions = torch.cat((padding, positions_morgans, padding), dim=0)

        mask = (1 - torch.eye(num_individuals)).bool() # is there some way we can make labels a pointer to predictions
        refs = SOI.unsqueeze(0).expand(num_individuals,-1,-1)[mask].reshape(num_individuals, num_individuals -1 , len_seq + input_size_positional - 1)
        labels = predictions.unsqueeze(0).expand(num_individuals,-1,-1,-1)[mask].reshape(num_individuals, num_individuals - 1, len_seq + input_size_positional - 1, num_classes)

        print("shapes")
        print(SOI.shape)
        print(refs.shape)
        print(predictions.shape)
        print(labels.shape)

        refs2 = torch.zeros(49, 48, 950).to(device)
        for i in range(49):
            refs2[i] = torch.cat((SOI[:i], SOI[i+1:]), dim=0)
        print(torch.equal(refs, refs2))

        strict_predictions = predictions.clone()
            
        when_predicted = torch.full((num_individuals, len_seq), 1.0)
        has_predicted = torch.zeros((num_individuals, len_seq))

        for i in range(0, num_individuals * len_seq, batch_size):
            probabilities = (when_predicted / when_predicted.sum()).flatten()
            ind = torch.multinomial(probabilities, batch_size, replacement=False)
            
            ind1 = ind // len_seq
            ind2 = ind % len_seq

            ind3, ind4 = ind1.clone(), ind2.clone() #############

            # ind1 = torch.randint(0, num_individuals, (batch_size,))
            # ind2 = torch.randint(0, len_seq, (batch_size,))

            has_predicted[ind1, ind2] = 1

            ind1 = ind1.unsqueeze(-1)
            ind2 = ind2.unsqueeze(-1) + torch.arange(input_size_positional).long()
            SOI_batch = SOI[ind1, ind2]
            refs_batch = refs[ind1, :, ind2].transpose(1,2)
            labels_batch = labels[ind1, :, ind2].transpose(1,2)
            positions_batch = positions.unsqueeze(0).expand(batch_size, -1)[torch.arange(batch_size).long().unsqueeze(-1), ind2]
            # torch.stack indices instead?  
            params_batch = torch.full((batch_size, 6), num_generations).to(device)

            out = self(SOI_batch, refs_batch, labels_batch, positions_batch, params_batch)
            out = F.softmax(out, dim=-1) # batch, num_classes

            # conider multiplying exp distribution by alpha?
            # len_exp_distribution = 49 ## this should be chosen based on num generations and threshold accuracy
            # positions_batch = (positions_batch[:, input_size_positional // 2 - len_exp_distribution // 2: input_size_positional // 2 + len_exp_distribution // 2 + 1] - positions_batch[:, input_size_positional // 2].unsqueeze(-1)).abs()
            # The below line is not exactly right.  Each predicted class could transition to another class with different probabilities
            # exp_distribution = torch.exp(-2 * num_generations * (positions_batch)) # batch, len_exp_distribution

            ### Calculate outside of loop
            lam = 2 * population_size * (1 - e ** (-num_generations / (2*population_size)))
            lam_a = admixture_proportion * lam
            lam_c = (1 - admixture_proportion) * lam
            transition_ancestry_probs = torch.tensor([[p_0a, 0, p_2a],
                                           [0, 1, 0],
                                           [p_2a, 0, p_0a]]).to(device)
            ###
            
            # add back in threshold value for longer sequences
            positions_diff = (positions - positions_batch[:, input_size_positional // 2].unsqueeze(-1)).abs().to(device) # batch, len_seq + input_size_positional
            transition_aa_haploid = lam_c / lam + (lam_a/lam) * torch.exp(-lam * positions_diff) # batch, len_seq + input_size_positional - 1
            transition_cc_haploid = lam_a / lam + (lam_c/lam) * torch.exp(-lam * positions_diff)
            transition_ac_haploid = 1 - transition_aa_haploid
            transition_ca_haploid = 1 - transition_cc_haploid
            
            transitions_aa_diploid = transition_aa_haploid ** 2
            transitions_ab_diploid = transition_aa_haploid * transition_ac_haploid * 2
            transitions_ac_diploid = transition_ac_haploid ** 2
            transitions_ba_diploid = transition_aa_haploid * transition_ca_haploid
            transitions_bb_diploid = transition_aa_haploid * transition_cc_haploid + transition_ac_haploid * transition_ca_haploid
            transitions_bc_diploid = transition_cc_haploid * transition_ac_haploid
            transitions_ca_diploid = transition_ca_haploid ** 2
            transitions_cb_diploid = transition_cc_haploid * transition_ca_haploid * 2
            transitions_cc_diploid = transition_cc_haploid ** 2

            transitions = torch.zeros((batch_size, len_seq + input_size_positional - 1, num_classes, num_classes)).to(device)
            transitions[:, :, 0, 0] = transitions_aa_diploid * p_0a + transitions_cc_diploid * (1 - p_0a)
            transitions[:, :, 0, 1] = transitions_ab_diploid * p_0a + transitions_cb_diploid * (1 - p_0a)
            transitions[:, :, 0, 2] = transitions_ac_diploid * p_0a + transitions_ca_diploid * (1 - p_0a)
            transitions[:, :, 1, 0] = transitions_ba_diploid * p_0a + transitions_bc_diploid * (1 - p_0a)
            transitions[:, :, 1, 1] = transitions_bb_diploid
            transitions[:, :, 1, 2] = transitions_bc_diploid * p_0a + transitions_ba_diploid * (1 - p_0a)
            transitions[:, :, 2, 0] = transitions_ca_diploid * p_0a + transitions_ac_diploid * (1 - p_0a)
            transitions[:, :, 2, 1] = transitions_cb_diploid * p_0a + transitions_ab_diploid * (1 - p_0a)
            transitions[:, :, 2, 2] = transitions_cc_diploid * p_0a + transitions_aa_diploid * (1 - p_0a)
            # transitions[:, :, 0, 0] = transition_aa_haploid ** 2
            # transitions[:, :, 0, 1] = transition_aa_haploid * transition_ac_haploid * 2
            # transitions[:, :, 0, 2] = transition_ac_haploid ** 2
            # transitions[:, :, 1, 0] = transition_aa_haploid * transition_ca_haploid
            # transitions[:, :, 1, 1] = transition_aa_haploid * transition_cc_haploid + transition_ac_haploid * transition_ca_haploid
            # transitions[:, :, 1, 2] = transition_cc_haploid * transition_ac_haploid
            # transitions[:, :, 2, 0] = transition_ca_haploid ** 2
            # transitions[:, :, 2, 1] = transition_cc_haploid * transition_ca_haploid * 2
            # transitions[:, :, 2, 2] = transition_cc_haploid ** 2

            out_smoothed = (out.unsqueeze(1).unsqueeze(1) @ transitions).squeeze(-2) #@ transition_ancestry_probs
            predictions[ind3] = predictions[ind3] * (1 - out_smoothed) + out_smoothed
            predictions[:, :input_size_positional // 2] = 0
            predictions[:, -(input_size_positional // 2):] = 0
            # print(out.shape)
            # print(transitions.shape)
            # print(out_smoothed.shape)
            # exit()

            ########
            # print(transitions[2,ind4[2]+250 + 5,:])
            # print(transitions[2,ind4[2]+250 + 250,:])
            # transitions[:, :, 0, 0] = transition_aa_haploid ** 2
            # transitions[:, :, 0, 1] = transition_aa_haploid * transition_ac_haploid * 2
            # transitions[:, :, 0, 2] = transition_ac_haploid ** 2
            # transitions[:, :, 1, 0] = transition_aa_haploid * transition_ca_haploid
            # transitions[:, :, 1, 1] = transition_aa_haploid * transition_cc_haploid + transition_ac_haploid * transition_ca_haploid
            # transitions[:, :, 1, 2] = transition_cc_haploid * transition_ac_haploid
            # transitions[:, :, 2, 0] = transition_ca_haploid ** 2
            # transitions[:, :, 2, 1] = transition_cc_haploid * transition_ca_haploid * 2
            # transitions[:, :, 2, 2] = transition_cc_haploid ** 2
            # print(transitions[2,ind4[2]+250 + 5,:])
            # print(transitions[2,ind4[2]+250 + 250,:])
            #######

            labels = predictions.unsqueeze(0).expand(num_individuals,-1,-1,-1)[mask].reshape(num_individuals, num_individuals - 1, len_seq + input_size_positional - 1, num_classes)

            #####
            strict_predictions[ind3, ind4] = out

            # if infer_num_generations and i % (batch_size * 100) == 0 and i > 0:

            #     num_tracts = (predictions[:, input_size_positional // 2: -(input_size_positional // 2) - 1].argmax(dim=-1) != predictions[:, input_size_positional // 2 + 1: -(input_size_positional // 2)].argmax(dim=-1)).sum().item() + num_individuals
            #     print(num_tracts)
            #     avg_len_transition = num_individuals * (5.8413e-02 - 5.1200e-06) / num_tracts
            #     print(avg_len_transition)

            #     num_generations = 1 / (4 * avg_len_transition)
            #     print(num_generations)
            #     print()

        return predictions[:, input_size_positional // 2: -(input_size_positional // 2)]


    @torch.no_grad()
    def predict_full_sequence(self, SOI, refs, labels, max_batch_size=batch_size):
        # SOI     #input_size_full
        # refs    #n_ind_max, input_size_full
        # labels  #n_ind_max, input_size_full

        assert SOI.shape[0] == refs.shape[1]
        assert refs.shape == labels.shape

        full_input_size = refs.shape[1]

        padding = torch.ones((input_size_positional // 2,)) * -1 
        SOI = torch.cat((padding, SOI, padding), dim=0)
        padding = torch.ones((n_ind_max, input_size_positional // 2)) * -1
        refs = torch.cat((padding, refs, padding), dim=-1)
        labels = torch.cat((padding, labels, padding), dim=-1)

        out = torch.zeros((full_input_size, num_classes)).to(device)
        for istart in range(0, full_input_size, max_batch_size):
            iend = min(istart + max_batch_size, full_input_size) 

            refs_batch = refs[:,istart:input_size_positional + iend - 1].to(device).unfold(-1, input_size_positional, 1).transpose(0, 1)
            labels_batch = labels[:,istart:input_size_positional + iend - 1].to(device).unfold(-1, input_size_positional, 1).transpose(0, 1)
            SOI_batch = SOI[istart:input_size_positional + iend - 1].to(device).unfold(0, input_size_positional, 1)

            out[istart:iend] = self(SOI_batch, refs_batch, labels_batch)

        return out

    def forward(self, SOI, refs, labels, positions, params):

        # print(SOI.shape, refs.shape, labels.shape)

        # SOI             # batch, input_size_positional
        # positions       # batch, input_size_positional
        # refs            # batch, n_ind_max, input_size_positional
        # labels          # batch, n_ind_max, input_size_positional, num_classses

        input_index, fit_index = fit_positions(positions, params)

        SOI_fitted = torch.full((SOI.shape[0], input_size_positional + 1), -1).to(device)
        SOI_gathered = torch.gather(SOI, 1, input_index)
        SOI_fitted.scatter_(1, fit_index, SOI_gathered) # push to device earlier
        SOI_fitted = SOI_fitted[:, :-1]

        positions_fitted = torch.full((positions.shape[0], input_size_positional + 1), 0).to(device).float()
        positions_gathered = torch.gather(positions, 1, input_index)
        positions_fitted.scatter_(1, fit_index, positions_gathered)
        positions_fitted = positions_fitted[:, :-1]

        input_index = input_index.unsqueeze(1).expand(-1, n_ind_max, -1)
        fit_index = fit_index.unsqueeze(1).expand(-1, n_ind_max, -1)

        refs_fitted = torch.full((refs.shape[0], n_ind_max, input_size_positional + 1), -1).to(device)
        refs_gathered = torch.gather(refs, 2, input_index)
        refs_fitted.scatter_(2, fit_index, refs_gathered)
        refs_fitted = refs_fitted[..., :-1]

        input_index = input_index.unsqueeze(-1).expand(-1, -1, -1, num_classes)
        fit_index = fit_index.unsqueeze(-1).expand(-1, -1, -1, num_classes)

        labels_fitted = torch.full((labels.shape[0], n_ind_max, input_size_positional + 1, num_classes), 0).to(device)
        labels_gathered = torch.gather(labels, 2, input_index)
        labels_fitted.scatter_(2, fit_index, labels_gathered)
        labels_fitted = labels_fitted[:, :, :-1]

        SOI = SOI_fitted
        positions = positions_fitted
        refs = refs_fitted
        labels = labels_fitted


        SOI = SOI.long().abs().unsqueeze(1).unsqueeze(1) # batch, 1, 1, input_size_positional

        idx = torch.randperm(num_classes * n_ind_pan_model)
        refs = refs.long()[:, idx]
        labels = labels[:, idx]

        # OHE distance with negative value encoding to all 0s
        mask1 = (refs < 0)
        refs = torch.abs(refs).unsqueeze(1) # batch, 1, n_ind_max, input_size_positional
        
        mask2 = (labels.sum(dim=-1) == 0)

        labels = torch.abs(labels).unsqueeze(1)        # batch, 1, n_ind_max, input_size_positional
        assert torch.equal(mask1, mask2)
        # print(mask1.sum().item()/mask1.numel())

        class_location = torch.arange(num_classes).long().unsqueeze(0).unsqueeze(-1).unsqueeze(-1).to(device) # batch, num_classes, 1, 1

        labels = labels.unsqueeze(4).unsqueeze(4).unsqueeze(4).expand(-1, -1, -1, -1, 3, 3, 3, -1)
        class_location = F.one_hot(class_location, num_classes=num_classes).unsqueeze(4).unsqueeze(4).unsqueeze(-1).expand(-1, -1, -1, -1, 3, 3, -1, 3)
        refs = F.one_hot(refs, num_classes=num_classes).unsqueeze(4).unsqueeze(-1).unsqueeze(-1).expand(-1, -1, -1, -1, 3, -1, 3, 3)
        SOI = F.one_hot(SOI, num_classes=num_classes).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand(-1, -1, -1, -1, -1, 3, 3, 3)

        ref_sim = labels * class_location * refs * SOI
        ref_sim = ref_sim.reshape(*ref_sim.shape[:4], -1).float() #batch, num_classes, n_ind_max, input_size_positional, num_classes ** 3

        ref_sim = ref_sim @ Transition.to(device) #batch, num_classes, n_ind_max, input_size_positional, n_embd
        
        ####
        # ref_sim_avg = ref_sim.mean(dim=-1, keepdim=True)
        # below line assumes constant recombination rate.  positions should be in morgans not bp
        positions = (positions - positions[:,input_size_positional // 2].unsqueeze(-1)).abs()
        admix_time = params[:,1].unsqueeze(-1)
        # pos_probs = (1 - positions) ** (2 * admix_time)
        # print(pos_probs.shape)
        # fix this ?? # assume large population size unless otherwise specified? 
        N = 1000 
        lam = 2 * N * (1 - torch.exp(-admix_time / (2 * N)))
        pos_probs = torch.exp(-lam * positions) # * 0.5 + 0.5
        # print(pos_probs.shape)
        # print(pos_probs[0, 200:300])
        # print()
        pos_probs = pos_probs.unsqueeze(1).unsqueeze(1).unsqueeze(-1).expand(-1, num_classes, n_ind_max, -1, -1)
        positions = positions.unsqueeze(1).unsqueeze(1).unsqueeze(-1).expand(-1, num_classes, n_ind_max, -1, -1)
        ref_sim = torch.cat((ref_sim, positions, pos_probs), dim=-1) #batch, num_classes, n_ind_max, input_size_positional, n_embd_model
        ####

        # Add noise
        # if self.training:
        #     ref_sim += torch.randn(ref_sim.shape).to(device) * sigma

        mask1 = mask1.unsqueeze(1).unsqueeze(-1).repeat(1, num_classes, 1, 1, n_embd_model)
        ref_sim[mask1] = 0


        # include position encoding here?
        # include frequency of each value here?

        # dist_avg = ref_sim.mean(dim=1, keepdim=True) # batch, 1, input_size_positional, 4
        # ref_sim = torch.cat((ref_sim, dist_avg), dim=1) # batch, num_classes * n_ind_pan_model + 1, input_size_positional, 4

        ref_sim = self.linear0(ref_sim)

        # final classification layers
        # ref_sim = self.block(ref_sim) ###
        ref_sim = ref_sim.reshape(*ref_sim.shape[:3], -1) # batch, num_classes, n_ind_max, input_size_positional * hidden0
        ref_sim = self.linear1(ref_sim)
        ref_sim = self.relu(ref_sim)
        ref_sim = self.linear2(ref_sim)
        ref_sim = self.relu(ref_sim)
        ref_sim = self.linear3(ref_sim) # batch, num_classes, num_classes * n_ind_pan_model

        # can pad inputs with less than 100 ref panels here instead of before to run faster
        # pad them with trained output of padded sequence
        # can encode class type (heterozygous or homozygous) here
        ref_sim, _ = torch.sort(ref_sim, dim=2) 
        ref_sim = self.sigmoid(ref_sim).squeeze(-1) # batch, num_classes, num_classes * n_ind_pan_model
        ref_sim = self.linear4(ref_sim)
        ref_sim = self.relu(ref_sim)
        ref_sim = self.linear5(ref_sim) # batch, num_classes, 1
        ref_sim = ref_sim.squeeze(-1)

        return ref_sim
    

def fit_positions(positions, params):

    ## can be calculated once outside of function?
    threshold_prob = 0.01
    admix_time = params[:,1]
    N = 1000 
    lam = 2 * N * (1 - torch.exp(-admix_time / (2 * N)))
    threshold_pos = -log(threshold_prob) / lam
    ##

    positions_fitted = torch.linspace(-1, 1, steps=input_size_positional).to(device)
    positions_fitted = threshold_pos.unsqueeze(-1) * positions_fitted # batch, input_size_positional

    positions_diff = (positions - positions[:, input_size // 2].unsqueeze(-1)) #batch, input_size

    cost_matrix = (positions_diff.unsqueeze(-1) - positions_fitted.unsqueeze(1)).abs() # batch, input_size, input_size_positional
    
    padding = torch.ones((positions_diff.shape[0], positions_diff.shape[1], positions_diff.shape[1])).to(device) * (threshold_pos / (input_size_positional // 2)).unsqueeze(-1).unsqueeze(-1)

    cost_matrix = torch.cat((cost_matrix, padding), dim=-1).cpu() # batch, input_size, input_size_positional + input_size

    input_index = []
    fit_index = []
    for i in range(cost_matrix.shape[0]):
        ind1, ind2 = linear_sum_assignment(cost_matrix[i])
        input_index.append(ind1)
        fit_index.append(ind2)

    input_index = torch.from_numpy(np.array(input_index)).to(device)
    fit_index = torch.from_numpy(np.array(fit_index)).to(device)
    mask = (fit_index >= input_size_positional)

    # input_index[mask] = input_size_positional
    fit_index[mask] = input_size_positional

    # fit_index = fit_index[0].tolist()
    # for i in range(input_size_positional):
    #     print(positions_fitted[0, i].item(), end="\t")
    #     print(positions_diff[0, fit_index.index(i)].item() if i in fit_index else "x")

    # exit()

    return input_index, fit_index


class TestNet(nn.Module):
    
    def __init__(self):
        super().__init__()

        self.linear = nn.Linear(input_size_positional, 100)
        self.linear2 = nn.Linear(100, 3)

        self.relu = nn.ReLU()

    def forward(self, SOI, refs, labels):

        SOI = self.linear(SOI.float())
        SOI = self.relu(SOI)
        SOI = self.linear2(SOI)
        return SOI

