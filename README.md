# Attention Mechanisms from Scratch

A clean PyTorch implementation of attention mechanisms — built step-by-step for learning and experimentation.

## Project Structure

```
├── self_attention.py   # Self-attention 
├── data.py             # Tokenization, embeddings, pipeline
├── masked_self_attention.py    # masked self-attention
├── multi_head_attention.py     # multi-head attention
└── README.md
```

## Quickstart

```bash
git clone https://github.com/Soham-47/attention-lab.git
cd attention-lab
pip install torch
python data.py
```

## How It Works

```
Sentence → Tokenize → Embed → (Q, K, V) → Scaled Dot-Product → Softmax → Output
```

Input shape: `(batch_size, seq_len, embed_dim)`

Masked self-attention applies a causal mask so each token only attends to itself and previous tokens — essential for autoregressive generation.

Multi-head attention divides the embedding into seperate heads so that the each attention head focuses on specific subspace of the embedding.

## References

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- [PyTorch Docs](https://pytorch.org/docs)


*Built for learning. Contributions welcome.*
