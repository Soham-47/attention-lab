import torch.nn as nn
import torch.nn.functional as F

class Self_Attention(nn.Module):
    """
    This is the self attention implementation on input shape (B , N , d)
    """
    def __init__(self , embed_dim , dim_key , dim_value):
        super().__init__()

        self.dim_key = dim_key

        self.W_q = nn.Linear(embed_dim, dim_key)
        self.W_k = nn.Linear(embed_dim, dim_key)
        self.W_v = nn.Linear(embed_dim , dim_value)
        self.W_o = nn.Linear(dim_value, embed_dim)

    def forward(self , x ):

        #x : (batch , seq_len , embed_dim)

        Q = self.W_q(x)
        K = self.W_k(x)
        V = self.W_v(x)

        scores = Q @ K.transpose(-2 , -1)

        scores = scores / (self.dim_key**0.5)

        weights = F.softmax(scores , dim = -1)

        output = weights @ V

        output = self.W_o(output)

        return output

