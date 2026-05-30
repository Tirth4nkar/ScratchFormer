<div align="center">

# 🤖 Scratchformer

### Building Transformers From First Principles

*A ground-up implementation of the Transformer Encoder-Decoder architecture to understand how modern language models actually work.*

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)]()
[![PyTorch](https://img.shields.io/badge/PyTorch-Learning-red.svg)]()
[![License](https://img.shields.io/badge/License-MIT-green.svg)]()

---

<img src=".\references\attention_research_1-727x1024.png" width="100%" alt="Transformer Architecture"/>

*Original Transformer architecture introduced in* **Attention Is All You Need (Vaswani et al., 2017)**

---

</div>

## 🎯 Project Goal

Most Transformer tutorials stop at:

```python
from transformers import AutoModel

model = AutoModel.from_pretrained(
    "google-t5/t5-small"
)
```

or

```python
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.2-1B"
)
```

Scratchformer takes the opposite approach.
Every major component is implemented and explored from the ground up:

- Token Embeddings
- Positional Encoding
- Scaled Dot-Product Attention
- Multi-Head Attention
- Residual Connections
- Layer Normalization
- Feed Forward Networks
- Encoder Blocks
- Decoder Blocks
- Masking Mechanisms
- Full Encoder-Decoder Architecture

The objective is not to build the fastest Transformer.
The objective is to understand **why Transformers work**.

---

## 🧠 What You'll Learn

This repository serves as a practical deep dive into the mechanics of modern language models.

By following the implementation you will understand:

### Attention

How queries, keys, and values interact to determine contextual relevance.

### Multi-Head Attention

Why multiple attention heads learn different relationships simultaneously.

### Positional Encoding

How Transformers understand sequence order despite having no recurrence.

### Encoder Representations

How contextual token embeddings emerge through stacked self-attention layers.

### Decoder Generation

How causal masking enables autoregressive sequence generation.

### Sequence-to-Sequence Learning

How encoder and decoder stacks communicate through cross-attention.

---

## 🏗️ Architecture

```text
Input Tokens
      │
      ▼
Token Embeddings
      │
      ▼
Positional Encoding
      │
      ▼

┌─────────────────────┐
│     Encoder × N     │
│                     │
│ Multi-Head Attn     │
│ Add & Norm          │
│ Feed Forward        │
│ Add & Norm          │
└─────────────────────┘

      │
      ▼

┌─────────────────────┐
│     Decoder × N     │
│                     │
│ Masked Self Attn    │
│ Cross Attention     │
│ Feed Forward        │
└─────────────────────┘

      │
      ▼

 Linear Projection
      │
      ▼

 Softmax
      │
      ▼

 Output Tokens
```

---

## 📂 Project Structure

This repository follows the Cookiecutter Data Science structure.

```text
scratchformer/
│
├── data/
│   ├── raw/
│   ├── interim/
│   ├── processed/
│   └── external/
│
├── notebooks/
│   ├── 01_attention_basics.ipynb
│   ├── 02_multihead_attention.ipynb
│   ├── 03_encoder.ipynb
│   └── 04_decoder.ipynb
│
├── src/
│   ├── embeddings/
│   ├── attention/
│   ├── encoder/
│   ├── decoder/
│   ├── transformer/
│   └── training/
│
├── models/
│
├── reports/
│   └── figures/
│
├── tests/
│
├── requirements.txt
│
└── README.md
```

---

## 🚀 Running The Project

### Clone Repository

```bash
git clone https://github.com/<username>/scratchformer.git

cd scratchformer
```

### Create Environment

```bash
conda create -n scratchformer python=3.11

conda activate scratchformer
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Run Training

```bash
python src/training/train.py
```

---

## 🔬 Future Roadmap

- [ ] Tokenizer implementation
- [ ] Vocabulary build for tokenizer
- [ ] Embedding Layer with Positional Encoding
- [ ] Scaled Dot-Product Attention
- [ ] Multi-Head Attention
- [ ] Residual Connections
- [ ] Layer Normalization
- [ ] Feed Forward Networks
- [ ] Encoder Blocks
- [ ] Decoder Blocks
- [ ] Masking Mechanisms
- [ ] KV Cache Implementation for the Decoder
- [ ] Full Encoder-Decoder Architecture

---

## 💡 Why Scratchformer?

The modern AI ecosystem often treats Transformers as black boxes.
Scratchformer exists to make every matrix multiplication, every residual connection, and every attention score transparent and understandable.

If you've ever wondered:
> "What actually happens inside a Transformer?"

This repository is the answer.

---

<div align="center">

### ⭐ If this project helps you understand Transformers, consider giving it a star.

**Scratchformer**  
*No magic. Just matrices.*

</div>