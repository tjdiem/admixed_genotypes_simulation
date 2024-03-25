from globals import *

class Head(nn.Module):
  
   def __init__(self, head_size):
       super().__init__()
       self.key = nn.Linear(n_embd, head_size, bias=False)
       self.query = nn.Linear(n_embd, head_size, bias=False)
       self.value = nn.Linear(n_embd, head_size, bias=False)
       self.register_buffer("communication_matrix", torch.ones(input_size, input_size))
       self.dropout = nn.Dropout(dropout)

   def forward(self, x1, x2):
       # Input (batch, 2*input_size, n_embd)
       # Output (batch, 2*input_size, head_size)
       k = self.key(x2) # (batch, input_size, head_size)
       q = self.query(x1)  # (batch, input_size, head_size)
       W = q @ k.transpose(-2,-1) * k.shape[-1]**-0.5 # (batch, input_size, input_size)
       W = W.masked_fill(self.communication_matrix == 0, float('-inf')) # (batch, input_size, input_size)
       W = F.softmax(W, dim=-1)
       W = self.dropout(W)

       v = self.value(x2) # (batch, input_size, head_size)
       out = W @ v # (batch, input_size, head_size)
       return out
  
class MultiHead(nn.Module):

   def __init__(self,num_heads,head_size):
       super().__init__()
       self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)]) #this can be parallelized
       self.linear = nn.Linear(head_size*num_heads,n_embd)
       self.dropout = nn.Dropout(dropout)

   def forward(self, x1, x2):
       x = torch.cat([head(x1, x2) for head in self.heads],dim=-1) #(batch,input_size,head_size (global))
       x = self.linear(x) #(batch,input_size,n_embd)
       x = self.dropout(x)
       return x
  
class FeedForward(nn.Module):

   def __init__(self, n_embd):
       super().__init__()
       self.net = nn.Sequential(
           nn.Linear(n_embd, 4 * n_embd),
           nn.ReLU(),
           nn.Linear(4 * n_embd, n_embd),
           nn.Dropout(dropout)
       )

   def forward(self, x):
       return self.net(x) #(batch,input_size,n_embd)
  
class EncoderDecoderBlock(nn.Module):

   def __init__(self):
       super().__init__()
       self.multihead1 = MultiHead(num_heads, head_size // num_heads)
       self.multihead2 = MultiHead(num_heads, head_size // num_heads)
       self.multihead3 = MultiHead(num_heads, head_size // num_heads)
       self.ffwd = FeedForward(n_embd)
       self.lns = nn.ModuleList([nn.LayerNorm(n_embd) for _ in range(6)])


   def forward(self, x1, x2):
        # input = output = (batch,input_size,n_embd)
        lnx1 = self.lns[0](x1)
        x1 = x1 + self.multihead1(lnx1, lnx1)

        lnx2 = self.lns[5](x2)
        x2 = x2 + self.multihead2(lnx2, lnx2)


        x1 = x1 + self.multihead3(self.lns[1](x1), self.lns[2](x2))

        x1 = x1 + self.ffwd(self.lns[3](x1))
        return x1, x2
   
class DecoderBlock(nn.Module):

   def __init__(self):
       super().__init__()
       self.multihead = MultiHead(num_heads, head_size // num_heads)
       self.ffwd = FeedForward(n_embd)
       self.ln1 = nn.LayerNorm(n_embd)
       self.ln2 = nn.LayerNorm(n_embd)


   def forward(self, x):
       # input = output = (batch,input_size,n_embd)
       lnx1 = self.ln1(x)
       x = x + self.multihead(lnx1, lnx1)
       x = x + self.ffwd(self.ln2(x))
       return x


class TransformerModel1(nn.Module):
    # outputs entire sequence

    def __init__(self):
        super().__init__()
        self.pos_embedding = nn.Embedding(input_size, n_embd)
        self.blocks = nn.ModuleList([DecoderBlock() for _ in range(num_blocks)])
        self.multihead = MultiHead(num_heads, head_size // num_heads)

        self.ln0 = nn.LayerNorm(n_embd)
        self.ln1 = nn.LayerNorm(input_size*n_embd)
        self.ln2 = nn.LayerNorm(input_size*n_embd//4)
        self.ln3 = nn.LayerNorm(input_size * n_embd // 16)

        self.linear1 = nn.Linear(input_size*n_embd,input_size*n_embd//4) #can change the output size of this
        self.linear2 = nn.Linear(input_size * n_embd//4,input_size * n_embd // 16) #probably should change this
        self.linear3 = nn.Linear(input_size * n_embd // 16, input_size)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):

        # x sample of interest (batch, len_seq, aa_embd)

        # x2 reference genome (batch, len_seq, label_embd)

        for block in self.blocks:
            x = block(x)  #(batch, input_size, n_embd)

        x = x.reshape(x.shape[0], input_size*n_embd) #(batch, input_size*n_embd)

        x = self.ln1(x) #(batch,input_size*n_embd)
        x = self.linear1(x) #(batch, input_size * n_embd//4)
        x = self.dropout(x)
        x = self.relu(x)

        x = self.ln2(x)
        x = self.linear2(x) #(batch, input_size*n_embd//8) #add layernorms?
        x = self.dropout(x)
        x = self.relu(x)

        x = self.ln3(x)
        x = self.linear3(x) #(batch, input_size)
        x = self.sigmoid(x)

        return x
    

class TransformerModel2(nn.Module):
    # outputs one element at a time

    def __init__(self):
        super().__init__()

        hidden1 = input_size*n_embd // 50
        hidden2 = input_size // 15

        self.pos_embedding = nn.Embedding(input_size, n_embd)
        self.blocks = nn.ModuleList([DecoderBlock() for _ in range(num_blocks)])
        self.multihead = MultiHead(num_heads, head_size // num_heads)

        self.ln0 = nn.LayerNorm(n_embd)
        self.ln1 = nn.LayerNorm(input_size*n_embd)
        self.ln2 = nn.LayerNorm(hidden1)
        self.ln3 = nn.LayerNorm(hidden2)

        self.linear1 = nn.Linear(input_size*n_embd, hidden1) #can change the output size of this
        self.linear2 = nn.Linear(hidden1, hidden2) #probably should change this
        self.linear3 = nn.Linear(hidden2, num_classes)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, y):

        #### x sample of interest (batch, input_size, n_embd - 1)

        # all inputs: (batch, input_size)

        # y = (y + 1).long()
        # y = F.one_hot(y)
        # y = y.reshape(y.shape[0], y.shape[1], num_classes  * y.shape[2])

        x1 = torch.cat((x,y),dim=-1)

        # x1 = x1.transpose(-2, -1)

        for block in self.blocks:
            x1 = block(x1)
            # x1, x2 = block(x1, x2)  #(batch, input_size, n_embd)

        x = x1.reshape(x1.shape[0], input_size*n_embd) #(batch, input_size*n_embd)

        x = self.ln1(x) #(batch,input_size*n_embd)
        x = self.linear1(x) #(batch, hidden1)
        x = self.dropout(x)
        x = self.relu(x)

        x = self.ln2(x)
        x = self.linear2(x) #(batch, hidden2) #add layernorms?
        x = self.dropout(x)
        x = self.relu(x)

        x = self.ln3(x)
        x = self.linear3(x) #(batch, num_classes)

        return x

class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(input_size * n_embd, input_size)
        self.linear2 = nn.Linear(input_size, input_size // 4)
        self.linear3 = nn.Linear(input_size // 4, num_classes)

        self.relu = nn.ReLU()

    def forward(self, x, y):

        # x (batch, input_size, num_samples + 1) + 1 is for site encoding
        # y (batch, input_size, num_samples)
        # SOI (batch)

        # n_embd = (num_samples + 1) + (num_samples - 1)
        #                x                     y    

        # print(x)   
        # print(y)

        # for i in range(-1, 2):
        #     print((x == i).sum().item())

        # print()

        # x *= 0
        # y[:,:,1:] = 0

        # y = (y + 1).long()
        # y = F.one_hot(y, num_classes)
        # y = y.reshape(y.shape[0], y.shape[1], num_classes  * y.shape[2]).float()

        y[:,:,0] = 0

        x1 = torch.cat((x, y),dim=-1)

        x1 = x1.reshape(x1.shape[0], -1)

        x1 = self.linear1(x1)
        x1 = self.relu(x1)
        x1 = self.linear2(x1)
        x1 = self.relu(x1)
        x1 = self.linear3(x1)

        return x1
    
class KNet(nn.Module):
    def __init__(self):
        super().__init__()

        hidden1 = 400
        hidden2 = 100

        C_out = 30

        kernel_size1 = 40

        self.conv1 = nn.Conv2d(n_embd_processing - 1, C_out, (kernel_size1, 3 * num_classes))
        
        self.linear1 = nn.Linear(C_out * (input_size - kernel_size1 + 1), hidden1)
        self.linear2 = nn.Linear(hidden1, hidden2)
        self.linear3 = nn.Linear(hidden2, num_classes)

        self.relu = nn.ReLU()

    def forward(self, x, y):

        # x (batch, input_size, num_samples + 1) + 1 is for site encoding
        # y (batch, input_size, num_samples)
        # SOI (batch)

        # n_embd = (num_samples + 1) + (num_samples - 1)
        #                x                     y  

        SOI = x[:,:,0]  #batch, input_size
        x = x[:,:,1:-1]  #batch, input_size, n_embd_processing - 1
        y = y[:,:,1:]   #batch, input_size, n_embd_processing - 1

        SOI = SOI.unsqueeze(-1).repeat(n_embd_processing - 1)
        
        x1 = torch.cat((SOI - x, y),dim=-2)  #batch, input_size, 3, n_embd_processing - 1

        x1 = x1.permute(0, 3, 1, 2)  #batch, n_embd_processing - 1, input_size, 3

        x1 = F.one_hot(x1.long() + 1, num_classes).float() # batch, n_embd_processing -1, input_size, 3, num_classes

        # print(x1.shape)

        x1 = x1.reshape(x1.shape[0], x1.shape[1], x1.shape[2], -1) #batch, n_embd_processing - 1, input_size, 3 * num_classes

        x1 = self.conv1(x1)

        x1 = x1.reshape(x1.shape[0], -1)

        x1 = self.linear1(x1)
        x1 = self.relu(x1)
        x1 = self.linear2(x1)
        x1 = self.relu(x1)
        x1 = self.linear3(x1)

        return x1
    

class ConvNet(nn.Module):
    def __init__(self):
        super().__init__()

        hidden1 = 400
        hidden2 = 100

        hidden_dim = 3

        C_out = n_embd_processing - 1

        kernel_size1 = 1

        self.conv1 = nn.Conv2d(n_embd_processing - 1, C_out, (kernel_size1, 3 * num_classes), groups=C_out)
        
        self.linear1 = nn.Linear(C_out * (input_size - kernel_size1 + 1), hidden1)
        self.linear2 = nn.Linear(hidden1, hidden2)
        self.linear3 = nn.Linear(hidden2, num_classes)

        self.relu = nn.ReLU()

    def forward(self, x, y):

        # x (batch, input_size, num_samples + 1) + 1 is for site encoding
        # y (batch, input_size, num_samples)
        # SOI (batch)

        # n_embd = (num_samples + 1) + (num_samples - 1)
        #                x                     y  

        SOI = x[:,:,0]  #batch, input_size
        x = x[:,:,1:-1]  #batch, input_size, n_embd_processing - 1
        y = y[:,:,1:]   #batch, input_size, n_embd_processing - 1

        x = x.unsqueeze(-2) # batch, input_size, 1, n_embd_processing - 1
        y = y.unsqueeze(-2) # batch, input_size, 1, n_embd_processing - 1

        SOI = SOI.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, 1, y.shape[-1])  #batch, input_size, 1, n_embd_processing

        x1 = torch.cat((SOI, x, y),dim=-2)  #batch, input_size, 3, n_embd_processing - 1

        x1 = x1.permute(0, 3, 1, 2)  #batch, n_embd_processing - 1, input_size, 3

        x1 = F.one_hot(x1.long() + 1, num_classes).float() # batch, n_embd_processing -1, input_size, 3, num_classes
        x1 = x1.reshape(x1.shape[0], x1.shape[1], x1.shape[2], -1) #batch, n_embd_processing - 1, input_size, 3 * num_classes

        x1 = self.conv1(x1)

        x1 = x1.reshape(x1.shape[0], -1)

        x1 = self.linear1(x1)
        x1 = self.relu(x1)
        x1 = self.linear2(x1)
        x1 = self.relu(x1)
        x1 = self.linear3(x1)

        return x1
    

class ConvNet2(nn.Module):
    def __init__(self):
        super().__init__()

        hidden1 = 400
        hidden2 = 100

        self.window = 25

        self.alpha = nn.Parameter(torch.tensor(2 ** -0.5))
        self.beta = nn.Parameter(torch.tensor(-(2 ** -0.5)))
        
        self.linear1 = nn.Linear(n_embd * (num_classes + 1), hidden1)
        self.linear2 = nn.Linear(hidden1, hidden2)
        self.linear3 = nn.Linear(hidden2, num_classes)

        self.relu = nn.ReLU()

        for name, param in self.state_dict().items():
            print(name, param.shape)

    def forward(self, x, y):

        # x (batch, input_size, num_samples + 1) + 1 is for site encoding
        # y (batch, input_size, num_samples)
        # SOI (batch)

        # n_embd = (num_samples + 1) + (num_samples - 1)
        #                x                     y  

        SOI = x[:,:,0]  #batch, input_size
        pos = x[:,:,-1] #batch, input_size
        x = x[:,:,1:-1]  #batch, input_size, n_embd
        y = y[:,:,1:]   #batch, input_size, n_embd

        SOI = SOI.unsqueeze(-1)

        # x = x.unsqueeze(-2) # batch, input_size, 1, n_embd

        # SOI = SOI.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, 1, y.shape[-1])  #batch, input_size, 1, n_embd_processing

        # x1 = torch.cat((SOI, x),dim=-2)  #batch, input_size, 2, n_embd

        # x1 = x1.permute(0, 3, 1, 2)  #batch, n_embd, input_size, 2

        # x = x.permute(2, 0, 1) # n_embd, batch, input_size

        x1 = (x * self.alpha + SOI * self.beta)#.permute(1, 2, 0)
        x1 = x1.transpose(-2,-1)
        x1 = x1.unsqueeze(-1)

        # print(x1.shape)

        y = y.transpose(-2, -1) # batch, n_embd, input_size
        y = F.one_hot(y.long() + 1, num_classes).float() #batch, n_embd, input_size, num_classes

        # print(y.shape)
        x1 = torch.cat((x1 ** 2, y),dim=-1) #batch, n_embd, input_size, num_classes + 1
        x1 = x1[:,:,input_size // 2 - self.window // 2: input_size // 2 + self.window // 2 + 1,:].mean(dim=2)  #batch, n_embd_procesing - 1, num_classes + 1

        for i in range(x1.shape[0]):
            indices = torch.argsort(x1[i,:,0])
            x1[i] = x1[i, indices]


        x1 = x1.reshape(x1.shape[0], -1)

        x1 = self.linear1(x1)
        x1 = self.relu(x1)
        x1 = self.linear2(x1)
        x1 = self.relu(x1)
        x1 = self.linear3(x1)

        return x1

class SimpleTest:

    def __init__(self):
        pass

    def eval(self):
        pass
    def test(self):
        pass

    def parameters(self):
        return [torch.tensor(1)]


    def __call__(self, x, y):

        y = y.long() + 1

        out = torch.zeros((x.shape[0],3))
        out2 = torch.zeros([x.shape[0],3])
        for i in range(x.shape[0]):
            allele = x[i, y.shape[1] // 2, 0]
            for j in range(1, y.shape[2]):
                if allele == x[i , y.shape[1] // 2, j]:
                    out[i][y[i, y.shape[1] // 2, j]] += 1
                else:
                    out2[i][y[i, y.shape[1] // 2, j]] += 1

        

        return out
                
class KNN:

    def __init__(self):
        pass

    def eval(self):
        pass

    def test(self):
        pass

    def train(self):
        pass

    def parameters(self):
        return [torch.tensor(1)]


    def __call__(self, x, y):

        window = 25
        threshold = 18
        k = 5

        def distance(arr1, arr2):
            return ((arr1 - arr2) ** 2).sum().item()

        y = y.long() + 1

        out = torch.zeros((x.shape[0], 3))
        for i in range(x.shape[0]):
            values = []
            allele = x[i, y.shape[1] // 2 - window // 2: y.shape[1] // 2 + window // 2 + 1, 0]
            for j in range(1, y.shape[2]):
                v = y[i , y.shape[1] // 2, j].item()
                if (y[i , y.shape[1] // 2 - window // 2: y.shape[1] // 2 + window // 2 + 1, j] == v).sum().item() >= threshold:
                    d = distance(x[i , y.shape[1] // 2 - window // 2: y.shape[1] // 2 + window // 2 + 1, j], allele)
                    values.append((d, v))

            values = sorted(values, key= lambda x: x[0])
            values = [v for _, v in values][:k]
            for a in range(3):
                out[i,a] = values.count(a)
            

        return out
    
"""
Actual model improvement:
layernorm setup in heads (layernorm after residual connection)
penalty for similar heads in multihead attention: probably won't work
For test examples we can input multiple permutations of each example and average the results: will probably help slightly
One hot encoding inputs: probably isn't necessary
distance between sites, plus other information, as input
custom weight initialization

"""
