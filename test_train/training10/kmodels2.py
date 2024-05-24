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

class KNet4(nn.Module):
    def __init__(self):
        super().__init__()

        hidden1 = 1000
        hidden2 = 100

        hidden3 = 50

        self.window = 25

        self.alpha = nn.Parameter(torch.tensor(2 ** -0.5))
        self.beta = nn.Parameter(torch.tensor(-(2 ** -0.5)))

        # self.block = DecoderBlock()

        self.linear1 = nn.Linear(input_size * n_embd, hidden1)
        self.linear2 = nn.Linear(hidden1, hidden2) 
        self.linear3 = nn.Linear(hidden2, 1)

        self.linear4 = nn.Linear((num_classes * n_ind_pan_model), hidden3)
        self.linear5 = nn.Linear(hidden3, 1)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

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

    def forward(self, SOI, refs, labels):

        # SOI             # batch, input_size
        # refs            # batch, n_ind_max, input_size
        # labels          # batch, n_ind_max, input_size, num_classses

        SOI = SOI.long().unsqueeze(1).unsqueeze(1) # batch, 1, 1, input_size

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

        # Add noise
        # if self.training:
        #     ref_sim += torch.randn(ref_sim.shape).to(device) * sigma

        mask1 = mask1.unsqueeze(1).unsqueeze(-1).repeat(1, num_classes, 1, 1, n_embd)
        ref_sim[mask1] = 0


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


"""
Knet3:
-----------------------------------------------------------------
Started training on epoch 1 of 1600, learning rate 0.0003000
Total time elapsed: 0h 0m 18s
-----------------------------------------------------------------
Started training on epoch 100 of 1600, learning rate 0.0002256
Total time elapsed: 0h 0m 32s
Loss train: 0.22217, Accuracy: 0.9180
Loss test: 0.18681, Accuracy: 0.9260
-----------------------------------------------------------------
Started training on epoch 200 of 1600, learning rate 0.0001691
Total time elapsed: 0h 0m 47s
Loss train: 0.17824, Accuracy: 0.9380
Loss test: 0.17273, Accuracy: 0.9360
-----------------------------------------------------------------
Started training on epoch 300 of 1600, learning rate 0.0001268
Total time elapsed: 0h 1m 1s
Loss train: 0.11879, Accuracy: 0.9580
Loss test: 0.18434, Accuracy: 0.9300
-----------------------------------------------------------------
Started training on epoch 400 of 1600, learning rate 0.0000951
Total time elapsed: 0h 1m 15s
Loss train: 0.12422, Accuracy: 0.9520
Loss test: 0.14224, Accuracy: 0.9440
-----------------------------------------------------------------
Started training on epoch 500 of 1600, learning rate 0.0000713
Total time elapsed: 0h 1m 30s
Loss train: 0.10991, Accuracy: 0.9560
Loss test: 0.13522, Accuracy: 0.9500


Knet4:
-----------------------------------------------------------------
Started training on epoch 1 of 1600, learning rate 0.0003000
Total time elapsed: 0h 0m 18s
-----------------------------------------------------------------
Started training on epoch 100 of 1600, learning rate 0.0002256
Total time elapsed: 0h 0m 56s
Loss train: 0.15247, Accuracy: 0.9480
Loss test: 0.11932, Accuracy: 0.9520
-----------------------------------------------------------------
Started training on epoch 200 of 1600, learning rate 0.0001691
Total time elapsed: 0h 1m 35s
Loss train: 0.12862, Accuracy: 0.9640
Loss test: 0.11122, Accuracy: 0.9600
-----------------------------------------------------------------
Started training on epoch 300 of 1600, learning rate 0.0001268
Total time elapsed: 0h 2m 14s
Loss train: 0.10028, Accuracy: 0.9600
Loss test: 0.10667, Accuracy: 0.9640
-----------------------------------------------------------------
Started training on epoch 400 of 1600, learning rate 0.0000951
Total time elapsed: 0h 2m 56s
Loss train: 0.08865, Accuracy: 0.9760
Loss test: 0.09609, Accuracy: 0.9720
-----------------------------------------------------------------
Started training on epoch 500 of 1600, learning rate 0.0000713
Total time elapsed: 0h 3m 39s
Loss train: 0.10497, Accuracy: 0.9580
Loss test: 0.10674, Accuracy: 0.9560
"""