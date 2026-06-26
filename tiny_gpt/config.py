class Config:
    data_path = "data/input.txt"
    checkpoint_path = "checkpoints/latest.pt"

    block_size = 128
    batch_size = 32

    n_embd = 192
    num_heads = 4
    num_layers = 4
    dropout = 0.2

    learning_rate = 1e-3
    train_steps = 5000
    save_every = 1000