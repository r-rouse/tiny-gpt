import os
import torch

from tiny_gpt.tokenizer import CharTokenizer
from tiny_gpt.data import TextDataset
from tiny_gpt.model import TinyGPT
from tiny_gpt.generate import generate


def main():
    text = open("data/input.txt", "r", encoding="utf-8").read()

    tokenizer = CharTokenizer(text)
    encoded = tokenizer.encode(text)

    block_size = 128
    checkpoint_file = "checkpoint.pt"

    dataset = TextDataset(
        encoded_text=encoded,
        block_size=block_size,
        batch_size=32,
    )

    model = TinyGPT(
        vocab_size=tokenizer.vocab_size,
        block_size=block_size,
        n_embd=192,
        num_heads=4,
        num_layers=4,
        dropout=0.2,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    start_step = 0

    if os.path.exists(checkpoint_file):
        checkpoint = torch.load(checkpoint_file)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_step = checkpoint["step"] + 1
        print(f"✅ Loaded checkpoint from step {checkpoint['step']}")
    else:
        print("🆕 Starting new model")

    for step in range(start_step, start_step + 5000):
        xb, yb = dataset.get_batch()

        logits, loss = model(xb, yb)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 500 == 0:
            print(f"step {step}: loss {loss.item():.4f}")

    context = torch.zeros((1, 1), dtype=torch.long)
    output = generate(model, context, max_new_tokens=500)
    print(tokenizer.decode(output[0].tolist()))

    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "step": step,
        "vocab_size": tokenizer.vocab_size,
        "block_size": block_size,
    }, checkpoint_file)

    print(f"✅ Saved checkpoint to {checkpoint_file}")


if __name__ == "__main__":
    main()