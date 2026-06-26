import os
import torch

from tiny_gpt.config import Config
from tiny_gpt.tokenizer import CharTokenizer
from tiny_gpt.data import TextDataset
from tiny_gpt.model import TinyGPT


def main():
    text = open(Config.data_path, "r", encoding="utf-8").read()

    tokenizer = CharTokenizer(text)
    encoded = tokenizer.encode(text)

    dataset = TextDataset(
        encoded_text=encoded,
        block_size=Config.block_size,
        batch_size=Config.batch_size,
    )

    model = TinyGPT(
        vocab_size=tokenizer.vocab_size,
        block_size=Config.block_size,
        n_embd=Config.n_embd,
        num_heads=Config.num_heads,
        num_layers=Config.num_layers,
        dropout=Config.dropout,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=Config.learning_rate)

    start_step = 0

    if os.path.exists(Config.checkpoint_path):
        checkpoint = torch.load(Config.checkpoint_path)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_step = checkpoint["step"] + 1
        print(f"✅ Loaded checkpoint from step {checkpoint['step']}")
    else:
        print("🆕 Starting new model")

    for step in range(start_step, start_step + Config.train_steps):
        xb, yb = dataset.get_batch()

        logits, loss = model(xb, yb)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 500 == 0:
            print(f"step {step}: loss {loss.item():.4f}")

        if step % Config.save_every == 0 and step > start_step:
            save_checkpoint(model, optimizer, tokenizer, step)

    save_checkpoint(model, optimizer, tokenizer, step)
    print(f"✅ Saved checkpoint to {Config.checkpoint_path}")


def save_checkpoint(model, optimizer, tokenizer, step):
    os.makedirs("checkpoints", exist_ok=True)

    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "step": step,
        "vocab_size": tokenizer.vocab_size,
        "block_size": Config.block_size,
    }, Config.checkpoint_path)


if __name__ == "__main__":
    main()