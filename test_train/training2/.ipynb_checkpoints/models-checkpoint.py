from globals import *

class Head(nn.Module):
  
   def __init__(self, head_size):
       super().__init__()
       self.key = nn.Linear(n_embd, head_size, bias=False)
       self.query = nn.Linear(n_embd, head_size, bias=False)
       self.value = nn.Linear(n_embd, head_size, bias=False)
       self.register_buffer("communication_matrix", torch.ones(input_size,input_size))
    #    self.communication_matrix[:input_size//2,:input_size//2] = 0
    #    self.communication_matrix[input_size//2:,input_size//2:] = 0
       self.dropout = nn.Dropout(dropout)

   def forward(self, x):
       # Input (batch, 2*input_size, n_embd)
       # Output (batch, 2*input_size, head_size)
       k = self.key(x) # (batch, input_size, head_size)
       q = self.query(x)  # (batch, input_size, head_size)
       W = q @ k.transpose(-2,-1) * k.shape[-1]**-0.5 # (batch, input_size, input_size)
       W = W.masked_fill(self.communication_matrix == 0, float('-inf')) # (batch, input_size, input_size)
       W = F.softmax(W, dim=-1)
       W = self.dropout(W)


       v = self.value(x) # (batch, input_size, head_size)
       out = W @ v # (batch, input_size, head_size)
       return out
  
class MultiHead(nn.Module):

   def __init__(self,num_heads,head_size):
       super().__init__()
       self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)]) #this can be parallelized
       self.linear = nn.Linear(head_size*num_heads,n_embd)
       self.dropout = nn.Dropout(dropout)

   def forward(self, x):
       x = torch.cat([head(x) for head in self.heads],dim=-1) #(batch,input_size,head_size (global))
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
  
class Block(nn.Module):


   def __init__(self):
       super().__init__()
       self.multihead = MultiHead(num_heads, head_size // num_heads)
       self.ffwd = FeedForward(n_embd)
       self.ln1 = nn.LayerNorm(n_embd)
       self.ln2 = nn.LayerNorm(n_embd)


   def forward(self, x):
       # input = output = (batch,input_size,n_embd)
       x = x + self.multihead(self.ln1(x))
       x = x + self.ffwd(self.ln2(x))
       return x

class TransformerModel1(nn.Module):

    def __init__(self):
        super().__init__()
        self.pos_embedding = nn.Embedding(input_size, n_embd)
        self.blocks = nn.Sequential(*[Block() for _ in range(num_blocks)])
        self.multihead = MultiHead(num_heads, head_size // num_heads)
        self.linear1 = nn.Linear(input_size*n_embd,input_size*n_embd//4) #can change the output size of this
        self.ln1 = nn.LayerNorm(input_size*n_embd)


        ##
        self.ln2 = nn.LayerNorm(input_size*n_embd//4)

        self.ln3 = nn.LayerNorm(100)

        self.linear2 = nn.Linear(input_size * n_embd//4,100)
        self.linear3 = nn.Linear(100,1)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

        self.dropout = nn.Dropout(dropout)

    @torch.no_grad()
    def evaltest(self, x, num_test, max_batch_size):
        # for each example, we feed the model num_test combinations of the same example and average results
        # max_batch_size denotes the maximum batch size that the model can take in one forward pass
        # in each combination of an example, the samples are shuffled.  Later, we can randomly multiply by -1 or 1 or randomly switch order of two sites
        
        num_test_current = num_test
        y = torch.zeros((x.shape[0],)).to(device)
        
        while num_test_current > 0:
            num_test_iter = min(num_test_current, max_batch_size)
            batch_size = max_batch_size // num_test_iter
    
            for istart in range(0,x.shape[0],batch_size):
                iend = min(istart+batch_size,x.shape[0])
                x_example = x[istart:iend].to(device)
                x_example = x_example.repeat_interleave(num_test_iter,dim=0)
                x_example = torch.stack([row[:,torch.randperm(input_size)] for row in x_example])
                y_example = self(x_example).to(device)
                y_example = y_example.reshape(-1,num_test_iter)
                y_example = y_example.sum(dim=1)
                y[istart:iend] += y_example

            num_test_current -= num_test_iter

        y /= num_test
        
        return y

    def forward(self, x):

        # X (batch, input_size, num_chrom)
        #print(x.shape)
    #    pos_embd = self.pos_embedding(torch.arange(input_size.to(device)) # (input_size, n_embd)
    #    x = x + pos_embd #(batch, input_size, n_embd) # we possibly want to concatenate this instead of adding
        #x = torch.cat((x, torch.zeros(len_chrom)),dim=2)

        # site_pos = (torch.arange(input_size)/input_size).to(device)
        # site_pos[-1] = -1
        # site_pos = site_pos.unsqueeze(1).unsqueeze(0)
        # site_pos = site_pos.repeat(x.shape[0],1,1)
        # x = torch.cat((x,site_pos),2)

        x = x.transpose(-2,-1)

        x = self.blocks(x) #(batch, input_size, n_embd)
        x = x.reshape(x.shape[0], input_size*n_embd) #(batch, input_size*n_embd)

        x = self.ln1(x) #(batch,input_size*n_embd)
        x = self.linear1(x) #(batch, input_size * n_embd//4)
        x = self.dropout(x)
        x = self.relu(x)

        x = self.ln2(x)
        x = self.linear2(x) #(batch, 100) #add layernorms?
        x = self.dropout(x)
        x = self.relu(x)

        x = self.ln3(x)
        x = self.linear3(x) #(batch, 1)
        x = x.reshape(-1) #(batch)
        x = self.sigmoid(x) #(batch)

        return x
    
class SumModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(input_size, 500)
        self.linear2 = nn.Linear(500, 100)
        self.linear3 = nn.Linear(100, 1)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):

        x = x.float().mean(dim=2)
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        x = self.relu(x)
        x = self.linear3(x)
        x = x.reshape(-1)
        x = self.sigmoid(x)

        return x


"""
SumModel: best avg dist 0.1733
Transformer: best avg dist 0.1264

Improvement: 
we can see from train and test evals on early epochs that the model is overfitting
model only takes 500 samples per chromosome due to GPU memory constraints
transformer might do better with split points instead of samples - is this possible?



Actual model improvement:
penalty for similar heads in multihead attention: probably won't work
For test examples we can input multiple permutations of each example and average the results: will probably help slightly
One hot encoding inputs: probably isn't necessary
distance between sites, plus other information, as input
custom weight initialization

"""
