import torch


class TextDataset:
    def __init__(self, encoded_text, block_size=64, batch_size=32):
        self.data = torch.tensor(encoded_text, dtype=torch.long)
        self.block_size = block_size
        self.batch_size = batch_size

    def get_batch(self):
        ix = torch.randint(len(self.data) - self.block_size, (self.batch_size,))
        x = torch.stack([self.data[i:i+self.block_size] for i in ix])
        y = torch.stack([self.data[i+1:i+self.block_size+1] for i in ix])
        return x, y