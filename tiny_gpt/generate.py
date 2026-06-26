import torch
import torch.nn.functional as F


@torch.no_grad()
def generate(model, idx, max_new_tokens):
    model.eval()

    for _ in range(max_new_tokens):
        # Only feed the model the most recent block_size tokens
        idx_cond = idx[:, -model.block_size:]

        logits, _ = model(idx_cond)

        logits = logits[:, -1, :]
        probs = F.softmax(logits, dim=-1)

        next_idx = torch.multinomial(probs, num_samples=1)
        idx = torch.cat((idx, next_idx), dim=1)

    model.train()
    return idx