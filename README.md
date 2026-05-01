# Attention Mechanisms from Scratch

A clean PyTorch implementation of attention mechanisms — built step-by-step for learning and experimentation.

## What's Implemented

- Tokenization & vocabulary mapping
- Token embeddings
- Scaled dot-product attention
- Single-head self-attention (Q, K, V projections)
- **Masked self-attention**
- Modular, extensible design

## Project Structure

```
├── self_attention.py   # Self-attention 
├── data.py             # Tokenization, embeddings, pipeline
├── masked_self_attention.py       # masked self-attention
└── README.md
```

## Quickstart

```bash
git clone https://github.com/your-username/attention-from-scratch.git
cd attention-from-scratch
pip install torch
python data.py
```

## How It Works

```
Sentence → Tokenize → Embed → (Q, K, V) → Scaled Dot-Product → Softmax → Output
```

Input shape: `(batch_size, seq_len, embed_dim)`

Masked self-attention applies a causal mask so each token only attends to itself and previous tokens — essential for autoregressive generation.

## References

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- [PyTorch Docs](https://pytorch.org/docs)

## Roadmap

- [ ] Multi-head attention
- [ ] Positional encoding
- [ ] Transformer encoder block
- [ ] Attention map visualization

---

*Built for learning. Contributions welcome.*
