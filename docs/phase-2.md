# Phase 2 - Building a Transformer

## Goal

Replace the simple Bigram model with a Transformer.

This is the same family of architecture used by GPT.

---

## Why?

The Bigram model only knows:

Current Character

↓

Next Character

It cannot remember previous words.

Transformers solve this.

---

## New Components

### Token Embeddings

Instead of representing characters as IDs,

each character becomes a learned vector.

Example:

A

↓

[0.12, -0.84, ...]

---

### Positional Embeddings

Transformers must learn word order.

Without positional embeddings:

dog

god

would look identical.

---

### Self Attention

Every character can examine previous characters.

Example:

"The king said"

When predicting the next character,

the model can attend to:

The

king

said

instead of only:

said

---

### Feed Forward Networks

After attention,

each token is processed by a small neural network.

This lets the model learn more complex patterns.

---

### Transformer Blocks

Each block contains:

LayerNorm

↓

Self Attention

↓

Residual Connection

↓

Feed Forward

↓

Residual Connection

Stacking multiple blocks increases reasoning ability.

---

## Results

Loss decreased dramatically.

Bigram:

~2.5

Transformer:

Below 0.5

Generated text became significantly more coherent.

The model learned:

Character names

Stage directions

Dialogue formatting

Shakespearean style