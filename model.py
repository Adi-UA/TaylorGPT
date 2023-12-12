import torch
from torch import nn
from tqdm.auto import tqdm


class AttentionHead(nn.Module):
    def __init__(self, input_size, head_size, block_size, dropout):
        super().__init__()

        self.key = nn.Linear(input_size, head_size)
        self.query = nn.Linear(input_size, head_size)
        self.value = nn.Linear(input_size, head_size)
        self.dropout = nn.Dropout(dropout)

        self.register_buffer("mask", torch.tril(torch.ones(block_size, block_size)))
        self.register_buffer("head_size", torch.tensor(head_size))

    def forward(self, x):
        _, T, _ = x.shape
        k = self.key(x)
        q = self.query(x)
        value = self.value(x)

        w = k @ q.transpose(1, 2) / torch.sqrt(self.head_size)
        w = w.masked_fill(self.mask[:T, :T] == 0, float("-inf"))
        w = torch.softmax(w, dim=-1)
        w = self.dropout(w)

        out = w @ value

        return out


class MultiHeadAttention(nn.Module):
    def __init__(self, input_size, head_size, block_size, n_heads, dropout):
        super().__init__()

        self.heads = nn.ModuleList(
            [
                AttentionHead(input_size, head_size, block_size, dropout)
                for _ in range(n_heads)
            ]
        )

        self.projection = nn.Sequential(
            nn.Linear(n_heads * head_size, input_size),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        out = torch.cat([head(x) for head in self.heads], dim=-1)  # n_heads * head_size

        return self.projection(out)


class TransformerBlock(nn.Module):
    def __init__(self, input_size, head_size, block_size, n_heads, dropout):
        super().__init__()

        self.self_attention = MultiHeadAttention(
            input_size, head_size, block_size, n_heads, dropout
        )
        self.norm1 = nn.LayerNorm(input_size)
        self.feed_forward = nn.Sequential(
            nn.Linear(input_size, 4 * input_size),
            nn.ReLU(),
            nn.Linear(4 * input_size, input_size),
            nn.Dropout(dropout),
        )
        self.norm2 = nn.LayerNorm(input_size)

    def forward(self, x):
        out = self.self_attention(x) + x
        out = self.norm1(out)

        out = self.feed_forward(out) + out
        out = self.norm2(out)

        return out


class TransformerDecoderModel(nn.Module):
    def __init__(
        self, vocab_size, embed_size, head_size, block_size, n_heads, n_layers, dropout
    ):
        super().__init__()
        self.embeddings = nn.Embedding(vocab_size, embed_size)
        self.positional_embeddings = nn.Embedding(block_size, embed_size)
        self.norm = nn.LayerNorm(embed_size)
        self.blocks = nn.Sequential(
            *[
                TransformerBlock(embed_size, head_size, block_size, n_heads, dropout)
                for _ in range(n_layers)
            ]
        )
        self.norm2 = nn.LayerNorm(embed_size)
        self.model_head = nn.Linear(embed_size, vocab_size)
        self.register_buffer("block_size", torch.tensor(block_size))

    def forward(self, x):
        _, T = x.shape
        positions = torch.arange(T, device=x.device)
        x = self.embeddings(x) + self.positional_embeddings(positions)
        x = self.norm(x)
        x = self.blocks(x)
        x = self.norm2(x)
        return self.model_head(x)

    def generate(self, x, n_tokens):
        for _ in tqdm(range(n_tokens), desc="Generating text"):
            context = x[:, -self.block_size :]

            logits = self.forward(context)

            logits = logits[:, -1, :]
            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            x = torch.cat([x, next_token], dim=-1)

        return x
