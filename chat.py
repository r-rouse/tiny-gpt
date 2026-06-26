import sys
import torch

from tiny_gpt.config import Config
from tiny_gpt.tokenizer import CharTokenizer
from tiny_gpt.model import TinyGPT
from tiny_gpt.generate import generate


def main():
    if len(sys.argv) < 2:
        prompt = ""
    else:
        prompt = " ".join(sys.argv[1:])

    text = open(Config.data_path, "r", encoding="utf-8").read()
    tokenizer = CharTokenizer(text)

    checkpoint = torch.load(Config.checkpoint_path)

    model = TinyGPT(
        vocab_size=tokenizer.vocab_size,
        block_size=Config.block_size,
        n_embd=Config.n_embd,
        num_heads=Config.num_heads,
        num_layers=Config.num_layers,
        dropout=0.0,
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    if prompt:
        context = torch.tensor(
            [tokenizer.encode(prompt)],
            dtype=torch.long,
        )
    else:
        context = torch.zeros((1, 1), dtype=torch.long)

    output = generate(model, context, max_new_tokens=500)
    print(tokenizer.decode(output[0].tolist()))


if __name__ == "__main__":
    main()