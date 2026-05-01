import torch 
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        if embed_dim <= 0 or num_heads <= 0:
            raise ValueError("embed_dim and num_heads must be positive")
        if embed_dim % num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads")

        super().__init__()
        self.num_heads = num_heads
        self.embed_dim = embed_dim
        self.individual_dim = embed_dim // num_heads

        self.W_q = nn.Linear(embed_dim, embed_dim)
        self.W_k = nn.Linear(embed_dim, embed_dim)
        self.W_v = nn.Linear(embed_dim, embed_dim)
        self.W_o = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        batch_size, seq_len, _ = x.shape

        Q, K, V = self.W_q(x), self.W_k(x), self.W_v(x)

        def split_heads(t):
            t = t.view(batch_size, seq_len, self.num_heads, self.individual_dim)
            return t.transpose(1, 2)

        Q, K, V = split_heads(Q), split_heads(K), split_heads(V)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.individual_dim ** 0.5)
        weights = torch.softmax(scores, dim=-1)
        attended = torch.matmul(weights, V)

        attended = attended.transpose(1, 2).contiguous()
        attended = attended.view(batch_size, seq_len, self.embed_dim)

        return self.W_o(attended)


        

            

    