import torch 
import torch.nn as nn
import torch.nn.functional as F

class MaskedSelfAttention(nn.Module):
    def __init__(self , embed_dim , d_key, d_value):
        super().__init__()
        self.embed_dim = embed_dim
        self.d_key = d_key
        self.d_value = d_value

        self.W_q = nn.Linear(embed_dim , d_key ) 
        self.W_k = nn.Linear(embed_dim , d_key)
        self.W_v = nn.Linear(embed_dim , d_value)
        self.W_o = nn.Linear(d_value, embed_dim)

    def forward(self , x):

        Q = self.W_q(x)
        K = self.W_k(x)
        V = self.W_v(x)

        N = x.size(1)
        mask = torch.tril(torch.ones(N, N, device=x.device))
        mask = mask.unsqueeze(0)   

        scores = Q @ K.transpose(-2 , -1)
        scores = scores/(self.d_key ** 0.5)
        scores = scores.masked_fill(mask == 0, float('-inf'))
        weights = F.softmax(scores , dim = -1)

        outputs = weights @ V
        outputs = self.W_o(outputs)

        return outputs 
