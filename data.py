import torch
import torch.nn.functional as F
from self_attention import Self_Attention
from masked_self_attention import MaskedSelfAttention

sentence = "India is a diverse nation consisting of different languages and cultures"

dict = {s:i for i ,s in enumerate(sorted(sentence.split()))}
print(dict)

sentence_int = torch.tensor([dict[s] for s in sentence.split()])
#print(sentence_int)

torch.manual_seed(123)

#torch.nn.Embedding(num_embeddings , embedding_dim)
embed = torch.nn.Embedding(11,16) 

embedded_sentence = embed(sentence_int) #detaches the embeddings from the computation graph

#print(embedded_sentence)
#print(embedded_sentence.shape)

embedded_sentence = embedded_sentence.unsqueeze(0)
d_q , d_k , d_v = 24 , 24 , 28 

self_attn = Self_Attention(16 , 24 , 28)

self_attn_output = self_attn(embedded_sentence)

masked_self_attn = MaskedSelfAttention(16 , 24 , 28)

masked_attn_output = masked_self_attn(embedded_sentence)

print(masked_attn_output)




