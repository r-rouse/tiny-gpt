# Phase 3 - Persistent Models

## Goal

Separate training from inference.

Our model should become something we can save, reload, and chat with.

---

## Checkpointing

Instead of discarding the model after training,

we save its weights.

checkpoint.pt

contains:

Model weights

Optimizer state

Training step

Metadata

This file is the model's learned knowledge.

---

## Loading Checkpoints

Instead of starting from random weights,

we load the previous checkpoint.

Training now continues where it stopped.

Example:

Run 1

Step 0

↓

5000

↓

Save

Run 2

Load Step 5000

↓

10000

↓

Save

---

## Refactoring

We separated responsibilities.

train.py

Responsible for:

Training

Saving checkpoints

Updating weights

chat.py

Responsible for:

Loading checkpoints

Generating text

Never changes the weights

---

## Configuration

Hyperparameters moved into config.py.

Examples:

Block size

Embedding dimension

Learning rate

Layers

Heads

This makes experimentation much easier.

---

## Understanding the Model

At this point the project contains every major component of a GPT training pipeline:

Tokenizer

Transformer

Embeddings

Attention

Loss Function

Gradient Descent

Checkpointing

Autoregressive Generation

Although tiny, this project now mirrors the architecture of modern language models.

---

## Biggest Lesson

The Python code is not the intelligence.

The checkpoint file is.

The code describes the brain.

The checkpoint contains everything the model has learned.