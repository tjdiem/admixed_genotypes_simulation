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
        self.pos_embedding = nn.Embedding(input_size, n_embd)
        self.blocks = nn.ModuleList([DecoderBlock() for _ in range(num_blocks)])
        self.multihead = MultiHead(num_heads, head_size // num_heads)

        self.ln0 = nn.LayerNorm(n_embd)
        self.ln1 = nn.LayerNorm(input_size*n_embd)
        self.ln2 = nn.LayerNorm(input_size*n_embd // 4)
        self.ln3 = nn.LayerNorm(input_size // 2)

        self.linear1 = nn.Linear(input_size*n_embd,input_size*n_embd//4) #can change the output size of this
        self.linear2 = nn.Linear(input_size * n_embd//4,input_size // 2) #probably should change this
        self.linear3 = nn.Linear(input_size // 2, num_classes)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

        self.dropout = nn.Dropout(dropout)

    def forward(self, x1, x2, x3, x4):

        #### x sample of interest (batch, input_size, n_embd - 1)

        # x1 (batch, input_size, n_embd/2) input
        # x2 (batch) encoding of which site to predict
        # x3 (batch) encoding of which individual to predict
        # x4 (batch, input_size, n_embd/2 - 1) output info

        x = torch.empty((x1.shape[0], input_size, n_embd - 1)).to(device)

        x[:,:,0:n_embd:2] = x1
        x[:,:,1:n_embd - 1:2] = x4[:,:,:-1]

        # move individual of interest to index 0
        for i in range(x3.size(0)):  #do this in parallel
            row = x[i,:,x3[i]].unsqueeze(1)      
            x[i] = torch.cat((row, x[i,:,:x3[i]], x[i,:,x3[i]+1:]), dim=-1)

        x2 = F.one_hot(x2.long(), input_size) #(batch, input_size)
        x2 = x2.unsqueeze(-1).float() #(batch, input_size, 1)

        x = torch.cat((x, x2),dim=-1) #(batch, input_size, n_embd)

        for block in self.blocks:
            x = block(x)  #(batch, input_size, n_embd)

        x = x.reshape(x.shape[0], input_size*n_embd) #(batch, input_size*n_embd)

        x = self.ln1(x) #(batch,input_size*n_embd)
        x = self.linear1(x) #(batch, input_size * n_embd//4)
        x = self.dropout(x)
        x = self.relu(x)

        x = self.ln2(x)
        x = self.linear2(x) #(batch, input_size // 2) #add layernorms?
        x = self.dropout(x)
        x = self.relu(x)

        x = self.ln3(x)
        x = self.linear3(x) #(batch, num_classes)

        return x
    
"""
Actual model improvement:
layernorm setup in heads (layernorm after residual connection)
penalty for similar heads in multihead attention: probably won't work
For test examples we can input multiple permutations of each example and average the results: will probably help slightly
One hot encoding inputs: probably isn't necessary
distance between sites, plus other information, as input
custom weight initialization

"""
