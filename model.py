import torch
import torch.nn as nn

class GPTSmall(nn.Module):
    def __init__(self, vocab_size, dim=256, n_heads=4, n_layers=4, max_len=128):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, dim)
        self.pos_emb = nn.Parameter(torch.randn(1, max_len, dim))
        self.blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model=dim, nhead=n_heads)
            for _ in range(n_layers)
        ])
        self.ln = nn.LayerNorm(dim)
        self.fc = nn.Linear(dim, vocab_size)

    def forward(self, x):
        tok = self.token_emb(x)
        seq = tok + self.pos_emb[:, :x.size(1), :]
        for block in self.blocks:
            seq = block(seq)
        seq = self.ln(seq)
        return self.fc(seq)
