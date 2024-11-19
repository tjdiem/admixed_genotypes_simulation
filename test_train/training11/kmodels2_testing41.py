from globals import *
from math import e, sqrt, log

print(__file__)

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

class KNet4(nn.Module):
    def __init__(self):
        super().__init__()

        hidden0 = n_embd_model

        hidden1 = 250
        hidden2 = 100

        hidden3 = 50

        self.linear0 = nn.Linear(n_embd_model, hidden0)
        self.ln0 = nn.LayerNorm(hidden0)

        self.linear1 = nn.Linear(input_size * hidden0, hidden1)
        self.ln1 = nn.LayerNorm(hidden1)
        self.linear2 = nn.Linear(hidden1, hidden2) 
        self.ln2 = nn.LayerNorm(hidden2)
        self.linear3 = nn.Linear(hidden2, 1)
        self.ln3 = nn.LayerNorm(num_classes * n_ind_pan_model)

        self.linear4 = nn.Linear((num_classes * n_ind_pan_model), hidden3)
        self.ln4 = nn.LayerNorm(hidden3)
        self.linear5 = nn.Linear(hidden3, 1)

        self.linear1c = nn.Linear((num_classes * n_ind_pan_model),(num_classes * n_ind_pan_model))
        self.ln1c = nn.LayerNorm(num_classes * n_ind_pan_model)

        self.linear1d = nn.Linear((num_classes * n_ind_pan_model),(num_classes * n_ind_pan_model))
        self.ln1d = nn.LayerNorm(num_classes * n_ind_pan_model)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

        # self.Transition = Transition

    def forward(self, ref_sim):


        ref_sim = self.linear0(ref_sim)
        # ref_sim = self.ln0(ref_sim)
        # ref_sim = self.relu(ref_sim)

        # final classification layers
        # ref_sim = self.block(ref_sim) ###
        ref_sim = ref_sim.reshape(*ref_sim.shape[:3], -1) # batch, num_classes, n_ind_max, input_size * hidden0
        ref_sim = self.linear1(ref_sim)
        ref_sim = self.ln1(ref_sim)
        ref_sim = self.relu(ref_sim)

        # ref_sim = ref_sim.transpose(-2, -1)
        # ref_sim = self.linear1c(ref_sim)
        # ref_sim = self.ln1c(ref_sim)
        # ref_sim = ref_sim.transpose(-2, -1)

        ref_sim = self.linear2(ref_sim)
        ref_sim = self.ln2(ref_sim)
        # RELU

        # ref_sim = ref_sim.transpose(-2, -1)
        # ref_sim = self.linear1d(ref_sim)
        # ref_sim = self.ln1d(ref_sim)
        ref_sim = self.relu(ref_sim)
        # ref_sim = ref_sim.transpose(-2, -1)

        ref_sim = self.linear3(ref_sim) # batch, num_classes, num_classes * n_ind_pan_model

        # can pad inputs with less than 100 ref panels here instead of before to run faster
        # pad them with trained output of padded sequence
        # can encode class type (heterozygous or homozygous) here
        # ref_sim, _ = torch.sort(ref_sim, dim=2) 
        ref_sim = ref_sim.squeeze(-1)
        # ref_sim = self.ln3(ref_sim)
        # ref_sim = self.sigmoid(ref_sim) # batch, num_classes, num_classes * n_ind_pan_model

        # ref_sim = self.linear4(ref_sim)
        # ref_sim = self.ln4(ref_sim)
        # ref_sim = self.relu(ref_sim)
        # ref_sim = self.linear5(ref_sim) # batch, num_classes, 1
        # ref_sim = ref_sim.squeeze(-1)
        ref_sim = ref_sim.mean(dim=-1)

        return ref_sim
    
def get_ref_sim(SOI, refs, labels, positions, params):
        
    # return torch.zeros((16, 3, 48, input_size, n_embd_model)).float().to(device)

    # SOI             # batch, input_size
    # positions       # batch, input_size
    # refs            # batch, n_ind_max, input_size
    # labels          # batch, n_ind_max, input_size, num_classses

    assert ((positions.amax(dim=-1) - positions.abs().amin(dim=-1)) <= 1).all(), f"max: {positions.amax(dim=-1)}, min: {positions.abs().amin(dim=-1)}, diff: {positions.amax(dim=-1) - positions.abs().amin(dim=-1)}"

    SOI = SOI.to(torch.int8).abs().unsqueeze(1).unsqueeze(1) # batch, 1, 1, input_size

    idx = torch.randperm(num_classes * n_ind_pan_model)
    refs = refs.to(torch.int8)[:, idx]
    labels = labels[:, idx]

    # OHE distance with negative value encoding to all 0s
    mask1 = (refs < 0)

    refs = torch.abs(refs).unsqueeze(1) # batch, 1, n_ind_max, input_size
    
    mask2 = (labels.sum(dim=-1) == 0)

    labels = torch.abs(labels).unsqueeze(1)        # batch, 1, n_ind_max, input_size
    assert torch.equal(mask1, mask2)
    # print(mask1.sum().item()/mask1.numel())

    class_location = torch.arange(num_classes).to(torch.int8).unsqueeze(0).unsqueeze(-1).unsqueeze(-1).to(device) # batch, num_classes, 1, 1


    ref_sim = class_location + num_classes * refs + (num_classes ** 2) * SOI

    ref_sim = F.one_hot(ref_sim.long(), num_classes**3).float().unsqueeze(-1).repeat(1, 1, 1, 1, 1, num_classes)
    ref_sim *= labels.unsqueeze(-2)
    ref_sim = ref_sim.reshape(*ref_sim.shape[:-2], -1)

    # labels = labels.unsqueeze(4).unsqueeze(4).unsqueeze(4).expand(-1, -1, -1, -1, 3, 3, 3, -1)
    # class_location = F.one_hot(class_location, num_classes=num_classes).unsqueeze(4).unsqueeze(4).unsqueeze(-1).expand(-1, -1, -1, -1, 3, 3, -1, 3)
    # refs = F.one_hot(refs, num_classes=num_classes).unsqueeze(4).unsqueeze(-1).unsqueeze(-1).expand(-1, -1, -1, -1, 3, -1, 3, 3)
    # SOI = F.one_hot(SOI, num_classes=num_classes).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand(-1, -1, -1, -1, -1, 3, 3, 3)

    # GetCudaMemory()
    # ref_sim = labels * class_location * refs * SOI
    # ref_sim = ref_sim.reshape(*ref_sim.shape[:4], -1).float() #batch, num_classes, n_ind_max, input_size, num_classes ** 4
    # GetCudaMemory()

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
    ref_sim = torch.cat((ref_sim, positions, pos_probs), dim=-1) #batch, num_classes, n_ind_max, input_size, n_embd_model
    ####

    # Add noise
    # if self.training:
    #     ref_sim += torch.randn(ref_sim.shape).to(device) * sigma

    mask1 = mask1.unsqueeze(1).unsqueeze(-1).repeat(1, num_classes, 1, 1, n_embd_model)
    ref_sim[mask1] = 0

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



