# Blackboard Transformers

Teaching Transformers to Learn Algorithms via 2D Spatial Reasoning

## Overview

This project explores whether Transformers can learn to execute algorithms rather than merely pattern-match. Inspired by Chain-of-Thought (CoT) prompting, we extend reasoning to two dimensions: instead of generating linear text traces, our model manipulates symbols on a 2D "blackboard" grid—similar to how humans solve multi-digit arithmetic on paper.

**Key Question:** Can a Transformer learn generalizable algorithmic procedures by "showing its work" on a spatial canvas?

## Motivation

Modern Large Language Models excel at many tasks but reveal a fundamental weakness in algorithmic reasoning: they pattern-match rather than compute. When asked to add two 12-digit numbers, an LLM might guess based on training patterns rather than executing the step-by-step carry algorithm. A calculator that has learned the algorithm can solve exponentially many unseen problems; an LLM often cannot.

This project investigates whether explicit spatial reasoning on a 2D blackboard can help models:
- Learn true algorithms, not just empirical distributions
- Generalize to longer problem instances (e.g., train on 3-digit, test on 10-digit numbers)
- Produce interpretable intermediate reasoning steps
- Self-correct by validating and erasing mistakes

## Features

- **2D Blackboard Reasoning**: Models operate on a grid structure, placing symbols at specific spatial locations
- **Custom Transformer Architecture**:
  - 2D Rotary Positional Embeddings (RoPE) for spatial awareness
  - Support for both encoder-only and alternative architectures
- **BERT-style MLM Training**: Masked language modeling with protected input regions and random corruption
- **Distributed Training**: Multi-GPU support via PyTorch DDP
- **Two Algorithmic Tasks**:
  - **Multi-digit Addition**: Learn the carry algorithm with explicit intermediate steps
  - **Sequence Alignment**: Fill dynamic programming tables for edit distance computation
- **Automatic Mixed Precision**: Faster training with reduced memory footprint
- **Inference & Generalization Testing**: Evaluate on longer sequences than seen during training

## Installation

### Requirements

```bash
pip install torch torchvision numpy
```

**Optional:** Weights & Biases for experiment tracking
```bash
pip install wandb
```

### Hardware

- Multi-GPU setup recommended (code uses PyTorch DDP)
- Works on single GPU by adjusting launch commands
- CPU training possible but slow

## Quick Start

### Training on Multi-digit Addition

```bash
# Edit train.py to set: TASK = 'ADDITION'
torchrun --nproc_per_node=4 train.py
```

This will:
1. Generate 8,000 random 3-digit addition problems
2. Train a Transformer to predict board state transitions
3. Run inference on test problems
4. Test generalization on 4-digit numbers

### Training on Sequence Alignment

```bash
# Edit train.py to set: TASK = 'ALIGNMENT'
torchrun --nproc_per_node=4 train.py
```

This will:
1. Generate 20,000 random DNA sequence pairs (length 9)
2. Train model to fill dynamic programming table
3. Run inference showing stepwise table completion

### Single GPU Training

```bash
torchrun --nproc_per_node=1 train.py
```

## Architecture

### Blackboard Transformer

The core architecture consists of:

1. **Token Embeddings**: Character-level vocabulary (digits, operators, letters)
2. **2D Positional Encoding**:
   - **RoPE (Rotary Positional Embedding)**: Separate frequency encodings for row and column positions
   - Alternative: 2D sinusoidal embeddings
3. **Transformer Encoder**: Stack of attention layers with custom position-aware attention
4. **Output Head**: Linear projection to vocabulary logits for each grid cell

### Key Innovation: 2D RoPE

Traditional RoPE encodes 1D sequence position. We extend it to 2D grids by:
- Splitting the embedding dimension in half
- Applying row-wise rotation to first half, column-wise to second half
- Enabling relative spatial reasoning (up/down, left/right)

This allows the model to learn location-independent algorithms that generalize to different grid regions and sizes.

## Tasks

### Multi-digit Addition

**Grid Layout:**
```
      1          ← Carry row
    1 1 2        ← First number
  + 2 3 5        ← Second number
  -------
    3 4 7        ← Result (computed step-by-step)
```

**Training Details:**
- Grid Size: 5×8
- Training: 3-digit numbers (8K samples, 20 epochs)
- Batch Size: 128
- Learning Rate: 1e-4
- Protected Regions: Input numbers and operator symbols
- Generalization Test: 4-digit numbers

### Sequence Alignment (Edit Distance)

**Grid Layout:**
```
      A C G T    ← Sequence 1
  A 0 1 2 3 4
  C 1 1 1 2 3
  G 2 2 2 1 2    ← DP table
  T 3 3 3 2 1
```

**Training Details:**
- Sequences: DNA (A, C, G, T), length 9
- Grid Size: 12×12
- Training: 20K samples, 50 epochs
- Batch Size: 1024
- Learning Rate: 1e-3

## Training Strategy

### BERT-style Masked Language Modeling

We use a modified BERT-style objective:

1. **Primary Targets**: All cells that differ between input and target board (algorithm steps)
2. **Secondary Targets**: Random 30% of unchanged cells (for robustness)
3. **Corruption**: 50% of secondary targets are randomly corrupted before prediction
4. **Protected Mask**: Input regions are never corrupted (problem statement stays intact)

This encourages the model to:
- Learn to correct errors (from random corruption)
- Fill in missing computation steps
- Respect problem constraints

## Research Directions

After implementing the baseline, explore:

### 1. Generalization & Positional Encodings
- Compare absolute vs. RoPE vs. learned 2D encodings
- Test length generalization (train 3-digit → test 10-digit)
- Test location generalization (train top-left → test bottom-right)
- Design encodings for unbounded "infinite blackboard"

### 2. Architecture Comparisons
- 1D Chain-of-Thought vs. 2D Blackboard
- Encoder-only vs. decoder-only vs. encoder-decoder
- Different attention patterns (causal, bidirectional, sparse)

### 3. Multi-task & Complex Operations
- Joint training on addition + subtraction
- Extend to multiplication, division
- Non-arithmetic algorithms: Sudoku, sorting, graph traversal

### 4. Self-correction
- Train on corrupted boards with "eraser" actions
- Validate intermediate states before proceeding
- Measure robustness on long reasoning chains

### 5. Mechanistic Interpretability
- Analyze attention head specialization
- Identify circuits for carry propagation, column reading
- Visualize learned algorithm structure

## Project Structure

```
.
├── train.py                          # Main training script
├── blackboard_transformers.tex       # Research paper/proposal
├── refs.bib                          # Bibliography
└── README.md                         # This file
```

## Configuration

Key hyperparameters can be modified at the top of `train.py`:

```python
TASK = 'ADDITION'  # or 'ALIGNMENT'
NUM_DIGITS = 3     # For addition task
SEQ_LEN = 9        # For alignment task
D_MODEL = 128      # Embedding dimension
NHEAD = 8          # Attention heads
NUM_LAYERS = 6     # Transformer layers
```

## Output Example

### Addition Inference
```
--- Initial Problem ---

    112
  +235
  ----


--- Step 1 ---

    112
  +235
  ----
    7

--- Step 2 ---
   1
    112
  +235
  ----
   47

--- Final Board State ---
   1
    112
  +235
  ----
  347

Correct Answer: 347
```

## References

This project builds on:

- **Transformers**: Vaswani et al. "Attention Is All You Need" (NeurIPS 2017)
- **Chain-of-Thought**: Wei et al. "Chain-of-Thought Prompting Elicits Reasoning in Large Language Models" (2022)
- **Scratchpads**: Nye et al. "Show Your Work: Scratchpads for Intermediate Computation with Language Models" (2021)
- **RoPE**: Su et al. "RoFormer: Enhanced Transformer with Rotary Position Embedding" (2021)
- **Vision Transformers**: Dosovitskiy et al. "An Image is Worth 16x16 Words" (ICLR 2021)

## License

This is a research project. Please cite appropriately if you build upon this work.

## Author

Amir Joudaki
[amir.joudaki@inf.ethz.ch](mailto:amir.joudaki@inf.ethz.ch)

---

**Research Goal:** Can we build models that learn algorithms, not just patterns? This project takes a step toward interpretable, generalizable, and verifiable algorithmic reasoning in neural networks.
