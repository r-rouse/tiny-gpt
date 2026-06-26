# Phase 1 - Building the Simplest Language Model

## Goal

Understand the absolute fundamentals of language models.

Instead of using an existing LLM, we built the smallest possible neural network that can learn text.

---

## Concepts Learned

### Tokens

The computer cannot understand letters or words.

The tokenizer converts text into integers.

Example:

hello

↓

[7, 4, 11, 11, 14]

---

### Vocabulary

The model creates a list of every unique character in the training text.

Example:

a
b
c
...
z
space
.
,
!

Each character receives an integer ID.

---

### Training Data

The model learns one simple task:

Predict the next character.

Example:

Input:

hell

Target:

ello

The model never memorizes entire sentences directly.

It learns to predict one character at a time.

---

### Bigram Model

Our first model was a Bigram model.

It only learns:

Current Character → Next Character

Example:

t → h

q → u

This is extremely limited because it has no memory.

---

### Loss Function

We measure how wrong the predictions are using Cross Entropy Loss.

Large loss = poor predictions

Small loss = better predictions

---

### Gradient Descent

Every training step:

1. Predict
2. Measure error
3. Compute gradients
4. Update weights

This loop is how every modern neural network learns.

---

## Result

The model generated text that looked vaguely like English but had no long-term coherence.

Example:

"thourg hest..."

This proved the training pipeline worked.