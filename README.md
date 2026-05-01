
# Attention Mechanisms from Scratch (PyTorch)

![Image](https://images.openai.com/static-rsc-4/2sZsx_YXV-wZZI1J2fbqJF3701LjkACOd8dRpWg8c4w04PSE4HqXYbhPgCLRF8j3wUu2gj4RQp7oXvb2LeCafSZ7CXA8lH4TLJac65SlAtSw0nGKLBOIo6K29m-y4tQPAWET2yGbB6auvwGGZTGun8Jksib2CM5OQ1cdQlsqo8L63ID7-CyQd97kB1XB6upt?purpose=fullsize)

![Image](https://images.openai.com/static-rsc-4/-zbGMtmt5QlB9CJeTJZ1e3w0xUYwN6dDY7MX335xUQrTue9gH5NAq3jjG_WG9MeUYyYv15q2JfsTvwzITCNHp4RGfocJD3PPnFIG12rFIQNld2gjH2Is-t2QeJE23DlcX7YrtEfgbegjx_AUQZTkghk6VXNrO0SXIeRmMwxIYE9VjcqBH2c6ksmC8muSlVbO?purpose=fullsize)

![Image](https://images.openai.com/static-rsc-4/QrMRDEX8E8CuCzgoV_eYcNPRFRsizGW47HmgpgGqVNuUL8yrF9qTeMzn45L5vCMXnVtxL6BJjRzvJ-DMk637ss8megq0dIa46a9UD3Bz7y5ed-II0wZA2KSA7-opw0uFF7gnjNmlfL-Myi5bKmerVJTK_gU53Mo4EQeY0C1zIM1rjVW6xrZzd_xeAP9ynJs3?purpose=fullsize)

![Image](https://images.openai.com/static-rsc-4/U1G_8Xa01C3uE8cy1gjtSb4y3IkqZM_6eP0fE5fGxalz7YQwvaQg7CEY7EER5SDQhci5Jh6jvVYNrjH0RQiIfuwjEjIr6oesyd8iubb9nDQPVGmivcKQGbDf0YGnMyNAJeRErmAJJXJ7p_yFGZ8J6Jmd4kqafsb7bkXo2hj2SlXuQanMOEG-Ac9R-IJ5nLny?purpose=fullsize)

![Image](https://images.openai.com/static-rsc-4/E5iF1lUlG8_bxtA4KEGsVUI1zfwSUbJc8t8aJF0f1DXYICXQzTYlAd9_P-r9wtmiIBy3w-xeZbd133mpjzxSBuM5CT9AUCqEbQBqnZKq7oJ5l5nT3VdGwNJ3YW0j4RaOWCwfn6iXoMmYUK27SGdCVTy-QdCcqgjJeFHel2mCTcu_7nnBR-IYmb3Btp7sJVUE?purpose=fullsize)

## Overview

This project implements core attention mechanisms from scratch using PyTorch. 

The repository focuses on building attention step-by-step, starting from tokenization and embeddings, and progressing toward modular self-attention implementations.

---

## Features

* Tokenization and vocabulary mapping
* Embedding layer integration
* Scaled dot-product attention
* Self-attention (single-head) implementation
* Modular PyTorch design for extensibility
* Shape tracking and debugging support

---

## Project Structure

```bash
.
├── self_attention.py      # Self-attention module (Q, K, V projections)
├── data.py                # Tokenization, embedding, and testing pipeline
├── multi_head.py          # Multi-head attention (future)
└── README.md
```

---

## Installation

```bash
git clone https://github.com/your-username/attention-from-scratch.git
cd attention-from-scratch
pip install torch
```

---

## Usage

Run the example pipeline:

```bash
python data.py
```

Typical workflow:

```text
Sentence → Tokenization → Integer Encoding → Embedding → Attention → Contextual Output
```

---

## Mathematical Formulation

The attention mechanism is defined as:

[
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
]

Where:

* ( Q ) represents queries
* ( K ) represents keys
* ( V ) represents values
* ( d_k ) is the key dimension used for scaling

---

## Implementation Details

### Input Representation

The model expects input tensors of shape:

```text
(batch_size, sequence_length, embedding_dimension)
```

### Attention Pipeline

1. Project input embeddings into query, key, and value spaces using linear layers
2. Compute attention scores using dot product: ( QK^T )
3. Scale scores by ( \sqrt{d_k} )
4. Apply softmax to obtain attention weights
5. Compute weighted sum of values
6. Optionally project output back to embedding dimension

---

## Example Output

```text
Attention Weights:
[0.628, 0.0002, 0.0003, ..., 0.365, ...]

Interpretation:
The token assigns higher importance to a subset of tokens while ignoring others.
```

---

## Key Concepts

* Attention enables global interaction between tokens
* Each token dynamically aggregates information from others
* The mechanism is fully parallelizable
* It eliminates the sequential bottleneck of RNN-based models

---

## References

* Attention Is All You Need
* PyTorch Documentation

---

## Future Work

* Multi-head attention implementation
* Positional encoding
* Transformer encoder block
* Visualization of attention maps

---

## License

This project is intended for educational and research purposes.

---

