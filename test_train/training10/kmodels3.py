from globals import *

OHE_values = torch.arange(3**4).reshape(3, 3, 3, 3) # SOI value, ref value, class we are in, labeled class of ref
OHE_values[:, :, 2, :] = torch.flip(OHE_values[:, :, 0, :], dims=(2,))
unique_elements, inverse_indices = torch.unique(OHE_values, return_inverse=True)
OHE_values = torch.arange(len(unique_elements))[inverse_indices]
OHE_values = OHE_values.flatten()

n_embd = 54
assert n_embd == OHE_values.max() + 1

Transition = torch.zeros((num_classes**4, n_embd))
Transition[torch.arange(num_classes**4).long(), OHE_values] = 1

# how to index:
# labeled class + class we are in * 3 + ref_value * 9 + SOI value * 27
#    var1              var2                  var3           var4
    
# if var3_1 == var3_2, var4_1 == var4_2, var2_1 == 2 - var2_2 != 1, var1_1 == 2 - var1_2
    # then values should be same
# else they should be different

class KNet5(nn.Module):
    def __init__(self):
        super().__init__()

        hidden1 = 1000
        hidden2 = 100

        hidden3 = 50

        self.window = 25

        self.alpha = nn.Parameter(torch.tensor(2 ** -0.5))
        self.beta = nn.Parameter(torch.tensor(-(2 ** -0.5)))

        # self.block = DecoderBlock()

        self.linear1 = nn.Linear(input_size * (n_embd + 1), hidden1)
        self.linear2 = nn.Linear(hidden1, hidden2) 
        self.linear3 = nn.Linear(hidden2, 1)

        self.linear4 = nn.Linear((num_classes * n_ind_pan_model), hidden3)
        self.linear5 = nn.Linear(hidden3, 1)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    @torch.no_grad()
    def predict_cluster(self, SOI, batch_size=batch_size):

        num_individuals, len_seq = SOI.shape

        padding = torch.full((num_individuals, input_size // 2), -1).to(device)
        SOI = torch.cat((padding, SOI, padding), dim=1) # (num_individuals, len_seq + input_size - 1)
        
        # fill with 0.25, 0.5, 0.25 # fill with ancestry proporotion
        predictions = torch.full((num_individuals, len_seq, num_classes), 1/num_classes).to(device) # (num_individuals, len_seq, num_classes)

        ######
        # hard coded for 2 ancestries.  # make sure formula is correct
        # should be based on positions
        ### ChatGPT: The CDF, which gives the probability that a tract length is less than or equal to somve value L, is: F(L) = 1 - e^(-NL)
        num_generations = 20 
        predictions[0, :, :2] = torch.exp(torch.arange(len_seq) * (-num_generations/100)).unsqueeze(-1).repeat(1,2) * (1/6) + (1/3)
        predictions[0, :, 2] = 1 - predictions[0, :, 0] - predictions[0, :, 1]

        padding = torch.full((num_individuals, input_size // 2, num_classes), 0).to(device)
        predictions = torch.cat((padding, predictions, padding), dim=1) # (num_individuals, len_seq + input_size - 1, num_classes)

        mask = (1 - torch.eye(num_individuals)).bool() # is there some way we can make labels a pointer to predictions
        refs = SOI.unsqueeze(0).expand(num_individuals,-1,-1)[mask].reshape(num_individuals, num_individuals -1 , len_seq + input_size - 1)
        labels = predictions.unsqueeze(0).expand(num_individuals,-1,-1,-1)[mask].reshape(num_individuals, num_individuals - 1, len_seq + input_size - 1, num_classes)

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
        #         out = self(SOI[istart:iend, :input_size], refs[istart:iend, :, :input_size], labels[istart:iend, :, :input_size])
        #         predictions[istart:iend, input_size // 2] = F.softmax(out, dim=-1)
        #         # right now the forward method chooses 48 random ref panels

            
            
        #     # unnecessary if we can make labels a pointer to predictions
        #     # indent/unindent this
        #     labels = predictions.unsqueeze(0).expand(num_individuals,-1,-1,-1)[mask].reshape(num_individuals, num_individuals - 1, len_seq + input_size - 1, num_classes)
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
            ind2 = ind2.unsqueeze(-1) + torch.arange(input_size).long()
            out = self(SOI[ind1, ind2], refs[ind1, :, ind2].transpose(1,2), labels[ind1, :, ind2].transpose(1,2))
            out = F.softmax(out, dim=-1)
            # torch.stack indices instead?  

            s += (predictions[ind1[:,0], ind2[:,0] + input_size // 2] - out).abs().sum().item()
            predictions[ind1[:,0], ind2[:,0] + input_size // 2] = out

            labels = predictions.unsqueeze(0).expand(num_individuals,-1,-1,-1)[mask].reshape(num_individuals, num_individuals - 1, len_seq + input_size - 1, num_classes)

            # if i % 100 == 0:
            #     print(i/ (num_individuals * len_seq * 6))
            #     print(s/100)
            #     print(when_predicted)
            #     s = 0


        return predictions[:, input_size // 2: -(input_size // 2)]
    
    @torch.no_grad()
    def predict_cluster2(self, SOI, batch_size=batch_size, admixture_proportion=None, num_generations=None):

        if admixture_proportion is None:
            admixture_proportion = 0.5
            infer_admixture_proportion = True
        else:
            infer_admixture_proportion = False
        
        if num_generations is None:
            num_generations = 20 #### Change this default value
            infer_num_generations = True
        else:
            infer_num_generations = False

        num_individuals, len_seq = SOI.shape

        padding = torch.full((num_individuals, input_size // 2), -1).to(device)
        SOI = torch.cat((padding, SOI, padding), dim=1) # (num_individuals, len_seq + input_size - 1)
        
        # fill with 0.25, 0.5, 0.25 # fill with ancestry proporotion
        predictions = torch.full((num_individuals, len_seq, num_classes), 1/num_classes).to(device) # (num_individuals, len_seq, num_classes)

        predictions[...,[0,2]] = 0.25
        predictions[...,1] = 0.5
        predictions[0, :, 0] = torch.exp(torch.arange(len_seq) * (-num_generations/100)) * (1/4) + (1/4)
        predictions[0, :, 1] = 0.5
        predictions[0, :, 2] = 1 - predictions[0, :, 0] - predictions[0, :, 1]

        padding = torch.full((num_individuals, input_size // 2, num_classes), 0).to(device)
        predictions = torch.cat((padding, predictions, padding), dim=1) # (num_individuals, len_seq + input_size - 1, num_classes)

        mask = (1 - torch.eye(num_individuals)).bool() # is there some way we can make labels a pointer to predictions
        refs = SOI.unsqueeze(0).expand(num_individuals,-1,-1)[mask].reshape(num_individuals, num_individuals -1 , len_seq + input_size - 1)
        labels = predictions.unsqueeze(0).expand(num_individuals,-1,-1,-1)[mask].reshape(num_individuals, num_individuals - 1, len_seq + input_size - 1, num_classes)

        print("shapes")
        print(SOI.shape)
        print(refs.shape)
        print(predictions.shape)
        print(labels.shape)

        refs2 = torch.zeros(49, 48, 950).to(device)
        for i in range(49):
            refs2[i] = torch.cat((SOI[:i], SOI[i+1:]), dim=0)
        print(torch.equal(refs, refs2))
            
        when_predicted = torch.full((num_individuals, len_seq), 1.0)
        for i in range(0, num_individuals * len_seq, batch_size):
            probabilities = (when_predicted / when_predicted.sum()).flatten()
            ind = torch.multinomial(probabilities, batch_size, replacement=False)
            
            ind1 = ind // len_seq
            ind2 = ind % len_seq
            # ind1 = torch.randint(0, num_individuals, (batch_size,))
            # ind2 = torch.randint(0, len_seq, (batch_size,))

            when_predicted += 1
            # when_predicted[ind1, ind2] -= 1

            ind1 = ind1.unsqueeze(-1)
            ind2 = ind2.unsqueeze(-1) + torch.arange(input_size).long()
            out = self(SOI[ind1, ind2], refs[ind1, :, ind2].transpose(1,2), labels[ind1, :, ind2].transpose(1,2))
            out = F.softmax(out, dim=-1)
            # torch.stack indices instead?  

            # conider multiplying exp distribution by alpha?
            k = 0.01 # k is constant determined by admixture time
            exp_distribution = torch.exp(-k * (torch.arange(49) - 24).abs()).to(device)
            out_smoothed = out.unsqueeze(1) * exp_distribution.unsqueeze(-1)
            predictions_idx = torch.stack([predictions[ind1[j,0], ind2[j,0] + input_size // 2 - 24: ind2[j,0] + input_size // 2 + 25] for j in range(batch_size)])
            predictions_smoothed = predictions_idx * (1 - exp_distribution.unsqueeze(-1))
            # print("joy")
            # print(out_smoothed.shape)
            # print(out[0])
            # print(out_smoothed[0])
            # print(predictions_idx.shape)
            # print(predictions_idx[0])
            # exit()
            for j in range(batch_size):
                predictions[ind1[j,0], ind2[j,0] + input_size // 2 - 24: ind2[j,0] + input_size // 2 + 25] = out_smoothed[j] + predictions_smoothed[j]
            predictions[:,:input_size // 2] = 0
            predictions[:,-(input_size // 2): ] = 0

            # alpha = 0.01
            # for prediction, i1, i2 in zip(out, ind1, ind2):
            #     prediction[i1[0], i2[0] + input_size // 2 - 24: i2 + input_size // 2 + 25] = prediction[i1[0], i2[0] + input_size // 2 - 24: i2 + input_size // 2 + 25]

            #1 predictions[ind1[:,0], ind2[:,0] + input_size // 2] *= out
            #1 predictions[ind1[:,0], ind2[:,0] + input_size // 2] /= predictions[ind1[:,0], ind2[:,0] + input_size // 2].sum(dim=-1, keepdim=True)
            # predictions[ind1[:,0], ind2[:,0] + input_size // 2] = out 
            #2 predictions[ind1[:,0], ind2[:,0] + input_size // 2] = out * torch.tensor([0.25, 0.5, 0.25]).to(device)
            #2 predictions[ind1[:,0], ind2[:,0] + input_size // 2] /= predictions[ind1[:,0], ind2[:,0] + input_size // 2].sum(dim=-1, keepdim=True)





            labels = predictions.unsqueeze(0).expand(num_individuals,-1,-1,-1)[mask].reshape(num_individuals, num_individuals - 1, len_seq + input_size - 1, num_classes)


        return predictions[:, input_size // 2: -(input_size // 2)]

    @torch.no_grad()
    def predict_full_sequence(self, SOI, refs, labels, max_batch_size=batch_size):
        # SOI     #input_size_full
        # refs    #n_ind_max, input_size_full
        # labels  #n_ind_max, input_size_full

        assert SOI.shape[0] == refs.shape[1]
        assert refs.shape == labels.shape

        full_input_size = refs.shape[1]

        padding = torch.ones((input_size // 2,)) * -1 
        SOI = torch.cat((padding, SOI, padding), dim=0)
        padding = torch.ones((n_ind_max, input_size // 2)) * -1
        refs = torch.cat((padding, refs, padding), dim=-1)
        labels = torch.cat((padding, labels, padding), dim=-1)

        out = torch.zeros((full_input_size, num_classes)).to(device)
        for istart in range(0, full_input_size, max_batch_size):
            iend = min(istart + max_batch_size, full_input_size) 

            refs_batch = refs[:,istart:input_size + iend - 1].to(device).unfold(-1, input_size, 1).transpose(0, 1)
            labels_batch = labels[:,istart:input_size + iend - 1].to(device).unfold(-1, input_size, 1).transpose(0, 1)
            SOI_batch = SOI[istart:input_size + iend - 1].to(device).unfold(0, input_size, 1)

            out[istart:iend] = self(SOI_batch, refs_batch, labels_batch)

        return out


    def forward(self, SOI, refs, labels, positions):

        # print("forward")
        # print(SOI.shape, refs.shape, labels.shape)

        # SOI             # batch, input_size
        # positions       # batch, input_size
        # refs            # batch, n_ind_max, input_size
        # labels          # batch, n_ind_max, input_size, num_classses

        SOI = SOI.long().abs().unsqueeze(1).unsqueeze(1) # batch, 1, 1, input_size

        idx = torch.randperm(num_classes * n_ind_pan_model)
        refs = refs.long()[:, idx]
        labels = labels[:, idx]

        # OHE distance with negative value encoding to all 0s
        mask1 = (refs < 0)
        refs = torch.abs(refs).unsqueeze(1) # batch, 1, n_ind_max, input_size
        
        mask2 = (labels.sum(dim=-1) == 0)

        labels = torch.abs(labels).unsqueeze(1)        # batch, 1, n_ind_max, input_size
        assert torch.equal(mask1, mask2)
        # print(mask1.sum().item()/mask1.numel())

        class_location = torch.arange(num_classes).long().unsqueeze(0).unsqueeze(-1).unsqueeze(-1).to(device) # batch, num_classes, 1, 1

        labels = labels.unsqueeze(4).unsqueeze(4).unsqueeze(4).expand(-1, -1, -1, -1, 3, 3, 3, -1)
        class_location = F.one_hot(class_location, num_classes=num_classes).unsqueeze(4).unsqueeze(4).unsqueeze(-1).expand(-1, -1, -1, -1, 3, 3, -1, 3)
        refs = F.one_hot(refs, num_classes=num_classes).unsqueeze(4).unsqueeze(-1).unsqueeze(-1).expand(-1, -1, -1, -1, 3, -1, 3, 3)
        SOI = F.one_hot(SOI, num_classes=num_classes).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand(-1, -1, -1, -1, -1, 3, 3, 3)

        ref_sim = labels * class_location * refs * SOI
        ref_sim = ref_sim.reshape(*ref_sim.shape[:4], -1).float() #batch, num_classes, n_ind_max, input_size, num_classes ** 3

        ref_sim = ref_sim @ Transition.to(device) #batch, num_classes, n_ind_max, input_size, n_embd
        
        ####
        positions = positions.unsqueeze(1).unsqueeze(1).unsqueeze(-1).expand(-1, num_classes, n_ind_max, -1, -1)
        ref_sim = torch.cat((positions, ref_sim),dim=-1)
        ####

        # Add noise
        # if self.training:
        #     ref_sim += torch.randn(ref_sim.shape).to(device) * sigma

        mask1 = mask1.unsqueeze(1).unsqueeze(-1).repeat(1, num_classes, 1, 1, n_embd + 1)
        ref_sim[mask1] = 0

        ####
        ref_sim = self.block(ref_sim)
        ####

        # include position encoding here?
        # include frequency of each value here?

        # dist_avg = ref_sim.mean(dim=1, keepdim=True) # batch, 1, input_size, 4
        # ref_sim = torch.cat((ref_sim, dist_avg), dim=1) # batch, num_classes * n_ind_pan_model + 1, input_size, 4

        # final classification layers
        # ref_sim = self.block(ref_sim) ###
        ref_sim = ref_sim.reshape(ref_sim.shape[0], ref_sim.shape[1], ref_sim.shape[2], -1)
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
    

class TestNet(nn.Module):
    
    def __init__(self):
        super().__init__()

        self.linear = nn.Linear(input_size, 100)
        self.linear2 = nn.Linear(100, 3)

        self.relu = nn.ReLU()

    def forward(self, SOI, refs, labels):

        SOI = self.linear(SOI.float())
        SOI = self.relu(SOI)
        SOI = self.linear2(SOI)
        return SOI

