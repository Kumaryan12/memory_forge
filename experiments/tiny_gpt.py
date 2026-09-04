import math
import random

import torch
import torch.nn as nn
import torch.nn.functional as F


torch.manual_seed(42)
random.seed(42)


# ==================================================
# DATA
# ==================================================

text = """
the cat sat on the mat
the dog sat on the rug
the cat likes the mat
the dog likes the rug
the cat chased the dog
the dog chased the cat
the cat sat down
the dog sat down
"""


words = text.lower().split()

vocab = sorted(set(words))

stoi = {
    word: idx
    for idx, word in enumerate(vocab)
}

itos = {
    idx: word
    for word, idx in stoi.items()
}


encoded = torch.tensor(
    [stoi[word] for word in words],
    dtype=torch.long
)


block_size = 4


def get_batch(batch_size=8):

    inputs = []
    targets = []

    max_start = (
        len(encoded)
        - block_size
        - 1
    )

    for _ in range(batch_size):

        start = random.randint(
            0,
            max_start
        )

        chunk = encoded[
            start:start + block_size + 1
        ]

        inputs.append(
            chunk[:-1]
        )

        targets.append(
            chunk[1:]
        )

    return (
        torch.stack(inputs),
        torch.stack(targets)
    )


# ==================================================
# ATTENTION
# ==================================================

class CausalSelfAttention(nn.Module):

    def __init__(
        self,
        embedding_dim,
        num_heads
    ):
        super().__init__()

        assert (
            embedding_dim
            % num_heads
            == 0
        )

        self.embedding_dim = (
            embedding_dim
        )

        self.num_heads = (
            num_heads
        )

        self.head_dim = (
            embedding_dim
            // num_heads
        )

        self.q_proj = nn.Linear(
            embedding_dim,
            embedding_dim,
            bias=False
        )

        self.k_proj = nn.Linear(
            embedding_dim,
            embedding_dim,
            bias=False
        )

        self.v_proj = nn.Linear(
            embedding_dim,
            embedding_dim,
            bias=False
        )

        self.out_proj = nn.Linear(
            embedding_dim,
            embedding_dim,
            bias=False
        )

    def forward(self, x):

        B, T, C = x.shape

        Q = self.q_proj(x)
        K = self.k_proj(x)
        V = self.v_proj(x)

        Q = Q.view(
            B,
            T,
            self.num_heads,
            self.head_dim
        ).transpose(1, 2)

        K = K.view(
            B,
            T,
            self.num_heads,
            self.head_dim
        ).transpose(1, 2)

        V = V.view(
            B,
            T,
            self.num_heads,
            self.head_dim
        ).transpose(1, 2)

        scores = (
            Q @ K.transpose(-2, -1)
        ) / math.sqrt(
            self.head_dim
        )

        mask = torch.triu(
            torch.ones(
                T,
                T,
                dtype=torch.bool,
                device=x.device
            ),
            diagonal=1
        )

        scores = scores.masked_fill(
            mask,
            float("-inf")
        )

        weights = torch.softmax(
            scores,
            dim=-1
        )

        output = (
            weights @ V
        )

        output = (
            output
            .transpose(1, 2)
            .contiguous()
            .view(B, T, C)
        )

        return self.out_proj(
            output
        )


# ==================================================
# FEED-FORWARD
# ==================================================

class FeedForward(nn.Module):

    def __init__(
        self,
        embedding_dim
    ):
        super().__init__()

        hidden_dim = (
            4 * embedding_dim
        )

        self.net = nn.Sequential(
            nn.Linear(
                embedding_dim,
                hidden_dim
            ),

            nn.GELU(),

            nn.Linear(
                hidden_dim,
                embedding_dim
            )
        )

    def forward(self, x):

        return self.net(x)


# ==================================================
# TRANSFORMER BLOCK
# ==================================================

class TransformerBlock(nn.Module):

    def __init__(
        self,
        embedding_dim,
        num_heads
    ):
        super().__init__()

        self.norm1 = nn.LayerNorm(
            embedding_dim
        )

        self.attention = (
            CausalSelfAttention(
                embedding_dim,
                num_heads
            )
        )

        self.norm2 = nn.LayerNorm(
            embedding_dim
        )

        self.ffn = FeedForward(
            embedding_dim
        )

    def forward(self, x):

        x = (
            x
            + self.attention(
                self.norm1(x)
            )
        )

        x = (
            x
            + self.ffn(
                self.norm2(x)
            )
        )

        return x


# ==================================================
# TINY GPT
# ==================================================

class TinyGPT(nn.Module):

    def __init__(
        self,
        vocab_size,
        block_size,
        embedding_dim=64,
        num_heads=4,
        num_layers=2
    ):
        super().__init__()

        self.block_size = (
            block_size
        )

        self.token_embedding = (
            nn.Embedding(
                vocab_size,
                embedding_dim
            )
        )

        self.position_embedding = (
            nn.Embedding(
                block_size,
                embedding_dim
            )
        )

        self.blocks = (
            nn.ModuleList([
                TransformerBlock(
                    embedding_dim,
                    num_heads
                )
                for _ in range(
                    num_layers
                )
            ])
        )

        self.final_norm = (
            nn.LayerNorm(
                embedding_dim
            )
        )

        self.lm_head = nn.Linear(
            embedding_dim,
            vocab_size
        )

    def forward(
        self,
        idx,
        targets=None
    ):

        B, T = idx.shape

        token_embeddings = (
            self.token_embedding(idx)
        )

        positions = torch.arange(
            T,
            device=idx.device
        )

        position_embeddings = (
            self.position_embedding(
                positions
            )
        )

        x = (
            token_embeddings
            + position_embeddings
        )

        for block in self.blocks:
            x = block(x)

        x = self.final_norm(x)

        logits = self.lm_head(x)

        loss = None

        if targets is not None:

            B, T, V = (
                logits.shape
            )

            logits_flat = (
                logits.view(
                    B * T,
                    V
                )
            )

            targets_flat = (
                targets.view(
                    B * T
                )
            )

            loss = F.cross_entropy(
                logits_flat,
                targets_flat
            )

        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        idx,
        max_new_tokens
    ):

        for _ in range(
            max_new_tokens
        ):

            idx_cond = idx[
                :,
                -self.block_size:
            ]

            logits, _ = self(
                idx_cond
            )

            logits = logits[
                :,
                -1,
                :
            ]

            probs = torch.softmax(
                logits,
                dim=-1
            )

            next_token = (
                torch.multinomial(
                    probs,
                    num_samples=1
                )
            )

            idx = torch.cat(
                [
                    idx,
                    next_token
                ],
                dim=1
            )

        return idx


# ==================================================
# DEVICE
# ==================================================

device = torch.device(
    "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)


print(
    "Using device:",
    device
)


# ==================================================
# MODEL
# ==================================================

model = TinyGPT(
    vocab_size=len(vocab),
    block_size=block_size,
    embedding_dim=64,
    num_heads=4,
    num_layers=2
).to(device)


num_parameters = sum(
    p.numel()
    for p in model.parameters()
)


print(
    "Parameters:",
    f"{num_parameters:,}"
)


# ==================================================
# TRAINING
# ==================================================

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=3e-3
)


num_steps = 1000

batch_size = 16


model.train()


for step in range(num_steps):

    x, y = get_batch(
        batch_size
    )

    x = x.to(device)
    y = y.to(device)

    logits, loss = model(
        x,
        y
    )

    optimizer.zero_grad()

    loss.backward()

    optimizer.step()

    if (
        step % 100 == 0
        or step == num_steps - 1
    ):

        print(
            f"Step {step:4d} | "
            f"Loss: {loss.item():.4f}"
        )


# ==================================================
# GENERATION
# ==================================================

model.eval()


context = torch.tensor(
    [[stoi["the"]]],
    dtype=torch.long,
    device=device
)


generated = model.generate(
    context,
    max_new_tokens=15
)


generated_ids = (
    generated[0]
    .cpu()
    .tolist()
)


generated_words = [
    itos[idx]
    for idx in generated_ids
]


print("\nGenerated text:\n")

print(
    " ".join(
        generated_words
    )
)