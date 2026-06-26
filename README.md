# Tiny-GPT

A from-scratch implementation of a GPT-style language model built for learning, experimentation, and understanding how modern large language models work under the hood.

Rather than relying on high-level libraries or pre-trained models, this project incrementally builds the components of a transformer language model—from raw text preprocessing to autoregressive text generation. Each phase focuses on a single concept, making the project both an educational resource and a foundation for future research.

---

## Goals

* Learn how GPT-style language models work internally.
* Build every major component from first principles.
* Document each stage of development.
* Experiment with model architectures, training strategies, and inference techniques.
* Create a clean codebase that can continue growing into more advanced language models.

---

## Project Roadmap

### Phase 1 — Character-Level Language Model

* Character-level tokenizer
* Vocabulary creation
* Text encoding and decoding
* Context window generation
* Mini-batch sampling
* Simple neural language model
* Cross-entropy loss
* Text generation

---

### Phase 2 — Transformer Foundations

* Self-attention
* Multi-head attention
* Positional embeddings
* Feed-forward networks
* Residual connections
* Layer normalization
* Transformer blocks
* Training loop improvements

---

### Phase 3 — Training on Shakespeare

* Train a GPT model on Tiny Shakespeare
* Autoregressive text generation
* Temperature sampling
* Context management
* Model checkpoints
* Experimentation with hyperparameters

---

### Future Phases

Some planned topics include:

* Byte Pair Encoding (BPE)
* Subword tokenization
* Larger datasets
* Better sampling methods
* KV Cache
* Flash Attention
* Mixed precision training
* LoRA fine-tuning
* Instruction tuning
* RLHF concepts
* Retrieval-Augmented Generation (RAG)
* Quantization
* Distributed training
* Mixture of Experts (MoE)
* Vision Transformers
* Multimodal models

---

## Project Structure

```
tiny-gpt/
│
├── README.md
├── requirements.txt
├── train.py
├── chat.py
├── model.py
├── tokenizer.py
├── data/
├── checkpoints/
├── phases/
│   ├── phase-1.md
│   ├── phase-2.md
│   └── phase-3.md
└── ...
```

*(The structure will evolve as new phases are added.)*

---

## Example Output

```text
$ python chat.py "PUCK"

PUCK:
Lord, what fools these mortals be...
```

As the project progresses, the quality and coherence of generated text should continually improve.

---

## Learning Philosophy

The purpose of Tiny-GPT is not to compete with production models, but to understand **why** they work.

Every feature is implemented only after understanding the underlying mathematics and engineering decisions. The repository favors readability, documentation, and incremental development over optimization.

This project serves as both:

* a personal learning journal,
* and a reference implementation for anyone interested in transformer architectures.

---

## Resources

Inspired by work from:

* Andrej Karpathy
* *Attention Is All You Need*
* GPT-2 and GPT-3 papers
* Hugging Face Transformers
* nanoGPT

---

## Contributing

Suggestions, issues, and pull requests are welcome. If you're also learning how language models work, feel free to follow along or experiment with your own ideas.

---

## License

MIT License
