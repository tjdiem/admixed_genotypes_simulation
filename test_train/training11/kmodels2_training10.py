from globals import *
from math import e, sqrt, log

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

input_size = 501

class KNet4(nn.Module):
    def __init__(self):
        super().__init__()

        hidden0 = n_embd_model

        hidden1 = 1000
        hidden2 = 100

        hidden3 = 50

        self.linear0 = nn.Linear(n_embd_model, hidden0)

        self.linear1 = nn.Linear(input_size * hidden0, hidden1)
        self.linear2 = nn.Linear(hidden1, hidden2) 
        self.linear3 = nn.Linear(hidden2, 1)

        self.linear4 = nn.Linear((num_classes * n_ind_pan_model), hidden3)
        self.linear5 = nn.Linear(hidden3, 1)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, SOI, refs, labels, positions, params, return_grad=False):

        # print(SOI)
        # print(refs)
        # print(labels)
        # print(positions)
        # print(params)

        # print(SOI.shape, refs.shape, labels.shape)

        # SOI             # batch, input_size
        # positions       # batch, input_size
        # refs            # batch, n_ind_max, input_size
        # labels          # batch, n_ind_max, input_size, num_classses

        # torch.set_printoptions(threshold=1000)
        # print('\n\n\n')
        # print(SOI[0, ::2])
        # print(positions[0, ::2])
        # print(refs[0,0,::2])
        # print(labels[0, 0, ::2].sum(dim=-1))

        assert ((positions.amax(dim=-1) - positions.abs().amin(dim=-1)) < 1).all()

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

        if return_grad:
            labels = labels.float().unsqueeze(4).unsqueeze(4).unsqueeze(4).expand(-1, -1, -1, -1, 3, 3, 3, -1)
            class_location = F.one_hot(class_location, num_classes=num_classes).float().unsqueeze(4).unsqueeze(4).unsqueeze(-1).expand(-1, -1, -1, -1, 3, 3, -1, 3)
            refs = F.one_hot(refs, num_classes=num_classes).float().unsqueeze(4).unsqueeze(-1).unsqueeze(-1).expand(-1, -1, -1, -1, 3, -1, 3, 3)
            SOI_pre = F.one_hot(SOI, num_classes=num_classes).float()
            SOI_pre.requires_grad_(True)
            SOI = SOI_pre.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand(-1, -1, -1, -1, -1, 3, 3, 3)

        else:
            labels = labels.unsqueeze(4).unsqueeze(4).unsqueeze(4).expand(-1, -1, -1, -1, 3, 3, 3, -1)
            class_location = F.one_hot(class_location, num_classes=num_classes).unsqueeze(4).unsqueeze(4).unsqueeze(-1).expand(-1, -1, -1, -1, 3, 3, -1, 3)
            refs = F.one_hot(refs, num_classes=num_classes).unsqueeze(4).unsqueeze(-1).unsqueeze(-1).expand(-1, -1, -1, -1, 3, -1, 3, 3)
            SOI = F.one_hot(SOI, num_classes=num_classes).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand(-1, -1, -1, -1, -1, 3, 3, 3)


        ref_sim = labels * class_location * refs * SOI
        ref_sim = ref_sim.reshape(*ref_sim.shape[:4], -1).float() #batch, num_classes, n_ind_max, input_size, num_classes ** 3
        

        ref_sim = ref_sim @ Transition.to(device) #batch, num_classes, n_ind_max, input_size, n_embd
        
        ####
        # ref_sim_avg = ref_sim.mean(dim=-1, keepdim=True)
        # below line assumes constant recombination rate.  positions should be in morgans not bp
        positions = (positions - positions[:,input_size // 2].unsqueeze(-1)).abs()
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
        ref_sim = torch.cat((ref_sim, positions, pos_probs), dim=-1)#batch, num_classes, n_ind_max, input_size, n_embd_model
        # ref_sim[:, 0, :, 250:] = 1
        # a = torch.randint(0,2,(num_classes,n_ind_max,499,n_embd_model)).float().to(device)
        # ref_sim[0,:,:,2:] = a
        # ref_sim[1,:,:,:-2] = a
        # print("simulated")
        ####

        # Add noise
        # if self.training:
        #     ref_sim += torch.randn(ref_sim.shape).to(device) * sigma

        mask1 = mask1.unsqueeze(1).unsqueeze(-1).repeat(1, num_classes, 1, 1, n_embd_model)
        ref_sim[mask1] = 0

        # if return_grad:
            # ref_sim.requires_grad_(True)


        # include position encoding here?
        # include frequency of each value here?

        # dist_avg = ref_sim.mean(dim=1, keepdim=True) # batch, 1, input_size, 4
        # ref_sim = torch.cat((ref_sim, dist_avg), dim=1) # batch, num_classes * n_ind_pan_model + 1, input_size, 4

        out = self.linear0(ref_sim)
        # out = self.relu(out)

        # final classification layers
        # out = self.block(out) ###
        out = out.reshape(*out.shape[:3], -1) # batch, num_classes, n_ind_max, input_size * hidden0
        out = self.linear1(out)
        out = self.relu(out)
        out = self.linear2(out)
        out = self.relu(out)
        out = self.linear3(out) # batch, num_classes, num_classes * n_ind_pan_model

        # can pad inputs with less than 100 ref panels here instead of before to run faster
        # pad them with trained output of padded sequence
        # can encode class type (heterozygous or homozygous) here
        out, _ = torch.sort(out, dim=2) 
        out = self.sigmoid(out).squeeze(-1) # batch, num_classes, num_classes * n_ind_pan_model
        out = self.linear4(out)
        out = self.relu(out)
        out = self.linear5(out) # batch, num_classes, 1
        out = out.squeeze(-1)

        if return_grad:
            return SOI_pre, out

        return out
    

