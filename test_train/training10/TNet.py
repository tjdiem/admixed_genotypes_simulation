from globals import *

class Head(nn.Module):
  
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(input_size, head_size, bias=False)
        self.query = nn.Linear(input_size, head_size, bias=False)
        self.value = nn.Linear(input_size, head_size, bias=False)
        self.register_buffer("communication_matrix", torch.ones(1, 1))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x1, x2):
        # Input (batch, n_embd_model, input_size)
        # Output (batch, n_embd_model, input_size)
        k = self.key(x2) # (batch, n_embd_model, head_size)
        q = self.query(x1)  # (batch, n_embd_model, head_size)
        W = q @ k.transpose(-2,-1) * k.shape[-1]**-0.5 # (batch, n_embd_model, n_embd_model)
        W = W.masked_fill(self.communication_matrix == 0, float('-inf')) # (batch, n_embd_model, n_embd_model)  #unneeded
        W = F.softmax(W, dim=-1) #unneeded
        W = self.dropout(W)  #unneeded, do not use droput here
 
        v = self.value(x2) # (batch, n_embd_model, head_size)
        out = W @ v # (batch, n_embd_model, head_size)
        return out
   
class Head2(nn.Module):
  
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd_model, head_size, bias=False)
        self.query = nn.Linear(n_embd_model, head_size, bias=False)
        self.value = nn.Linear(n_embd_model, head_size, bias=False)
        self.register_buffer("communication_matrix", torch.ones(n_embd_model, n_embd_model))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x1, x2):
        # Input (batch, input_size, 1)
        # Output (batch, input_size, 1)
        k = self.key(x2) # (batch, input_size, head_size)
        q = self.query(x1)  # (batch, input_size, head_size)
        W = q @ k.transpose(-2,-1) * k.shape[-1]**-0.5 # (batch, input_size, input_size)
        # W = W.masked_fill(self.communication_matrix == 0, float('-inf')) # (batch, 1, 1)  #unneeded
        W = F.softmax(W, dim=-1) #unneeded
        W = self.dropout(W)  #unneeded, do not use droput here

        v = self.value(x2) # (batch, input_size, head_size)
        out = W @ v # (batch, input_size, head_size)
        return out
  
class MultiHead(nn.Module):

    def __init__(self,num_heads,head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)]) #this can be parallelized
        self.linear = nn.Linear(head_size*num_heads,input_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x1, x2):
        x1 = x1.transpose(-2, -1)
        x2 = x2.transpose(-2, -1)
        x = torch.cat([head(x1, x2) for head in self.heads],dim=-1) #(batch,n_embd_model,head_size (global))
        x = self.linear(x) #(batch,n_embd_model,input_size)
        x = self.dropout(x)
        x = x.transpose(-2, -1)
        return x
   
class MultiHead2(nn.Module):

    def __init__(self,num_heads,head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head2(head_size) for _ in range(num_heads)]) #this can be parallelized
        self.linear = nn.Linear(head_size*num_heads, n_embd_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x1, x2):
        # x1 = x1.transpose(-2, -1)
        # x2 = x2.transpose(-2, -1)
        x = torch.cat([head(x1, x2) for head in self.heads],dim=-1) #(batch,input_size,head_size (global))
        x = self.linear(x) #(batch,input_size,1)
        # x = x.transpose(-2, -1)
        x = self.dropout(x)
        return x
   
class FeedForward(nn.Module):

    def __init__(self, n_embd_model):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd_model, 4 * n_embd_model),
            nn.ReLU(),
            nn.Linear(4 * n_embd_model, n_embd_model),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x) #(batch,input_size,n_embd_model)
   
class DecoderBlock(nn.Module):

    def __init__(self):
        super().__init__()
        self.multihead = MultiHead2(num_heads, head_size // num_heads)
        self.ffwd = FeedForward(n_embd_model)
        self.ln1 = nn.LayerNorm(n_embd_model)
        self.ln2 = nn.LayerNorm(n_embd_model)
 

    def forward(self, x):
        # input = output = (batch,input_size,n_embd_model)
        lnx1 = self.ln1(x)
        x = x + self.multihead(lnx1, lnx1)
        x = x + self.ffwd(self.ln2(x))
        return x

