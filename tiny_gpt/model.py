import torch
import torch.nn as nn
import torch.nn.functional as F


class Head(nn.Module):
    def __init__(self, n_embd, head_size, block_size, dropout):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)

        self.register_buffer(
            "tril",
            torch.tril(torch.ones(block_size, block_size))
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        batch, time, channels = x.shape

        k = self.key(x)
        q = self.query(x)

        weights = q @ k.transpose(-2, -1) * channels ** -0.5
        weights = weights.masked_fill(
            self.tril[:time, :time] == 0,
            float("-inf")
        )

        weights = F.softmax(weights, dim=-1)
        weights = self.dropout(weights)

        v = self.value(x)
        out = weights @ v

        return out


class MultiHeadAttention(nn.Module):
    def __init__(self, n_embd, num_heads, head_size, block_size, dropout):
        super().__init__()
        self.heads = nn.ModuleList([
            Head(n_embd, head_size, block_size, dropout)
            for _ in range(num_heads)
        ])
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([head(x) for head in self.heads], dim=-1)
        out = self.proj(out)
        out = self.dropout(out)
        return out


class FeedForward(nn.Module):
    def __init__(self, n_embd, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class TransformerBlock(nn.Module):
    def __init__(self, n_embd, num_heads, block_size, dropout):
        super().__init__()

        head_size = n_embd // num_heads

        self.self_attention = MultiHeadAttention(
            n_embd=n_embd,
            num_heads=num_heads,
            head_size=head_size,
            block_size=block_size,
            dropout=dropout,
        )

        self.feed_forward = FeedForward(n_embd, dropout)

        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.self_attention(self.ln1(x))
        x = x + self.feed_forward(self.ln2(x))
        return x


class TinyGPT(nn.Module):
    def __init__(
        self,
        vocab_size,
        block_size=128,
        n_embd=128,
        num_heads=4,
        num_layers=4,
        dropout=0.2,
    ):
        super().__init__()

        self.block_size = block_size

        self.token_embedding = nn.Embedding(vocab_size, n_embd)
        self.position_embedding = nn.Embedding(block_size, n_embd)

        self.blocks = nn.Sequential(*[
            TransformerBlock(
                n_embd=n_embd,
                num_heads=num_heads,
                block_size=block_size,
                dropout=dropout,
            )
            for _ in range(num_layers)
        ])

        self.ln_final = nn.LayerNorm(n_embd)
        self.output_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        batch, time = idx.shape

        token_emb = self.token_embedding(idx)
        position_ids = torch.arange(time, device=idx.device)
        position_emb = self.position_embedding(position_ids)

        x = token_emb + position_emb
        x = self.blocks(x)
        x = self.ln_final(x)

        logits = self.output_head(x)

        loss = None
        if targets is not None:
            batch, time, channels = logits.shape
            logits_flat = logits.view(batch * time, channels)
            targets_flat = targets.view(batch * time)
            loss = F.cross_entropy(logits_flat, targets_flat)

        return logits, loss