# GPT Reproduction & Fine-Tuning from Scratch (PyTorch)

This project reproduces a **GPT-style language model from first principles** using PyTorch.  
The model implements **causal self-attention, multi-head attention, transformer blocks**, and is trained on **real-world text data** using a GPU.

The objective is **architectural understanding and hands-on experimentation**, not large-scale GPT replication.

---

## Project Objectives

- Reproduce the core architecture of GPT
- Implement causal (autoregressive) self-attention
- Experiment with key hyperparameters:
  - Number of layers
  - Number of attention heads
  - Embedding size
- Train and fine-tune the model on **real data**
- Generate autoregressive text outputs
- Analyze performance and tradeoffs

---

## Model Architecture

- **Architecture**: GPT-style Transformer Decoder
- **Tokenizer**: Character-level
- **Attention Type**: Causal multi-head self-attention
- **Framework**: PyTorch

### Final Configuration Used

```python
embedding_size = 256
n_heads = 4
n_layers = 4
block_size = 128
dropout = 0.1
