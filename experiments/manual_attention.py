import math

import torch
import torch.nn as nn


# --------------------------------------------------
# CONFIG
# --------------------------------------------------

torch.manual_seed(42)

tokens = [
    "the",
    "cat",
    "sat",
    "down",
]

token_ids = torch.tensor([
    0,
    1,
    2,
    3,
])

vocab_size = 4
embedding_dim = 8


# --------------------------------------------------
# EMBEDDING
# --------------------------------------------------

embedding = nn.Embedding(
    num_embeddings=vocab_size,
    embedding_dim=embedding_dim,
)

X = embedding(token_ids)

print("\nInput shape:")
print(X.shape)

print("\nEmbeddings:")
print(X)


# --------------------------------------------------
# Q, K, V PROJECTIONS
# --------------------------------------------------

W_q = nn.Linear(
    embedding_dim,
    embedding_dim,
    bias=False,
)

W_k = nn.Linear(
    embedding_dim,
    embedding_dim,
    bias=False,
)

W_v = nn.Linear(
    embedding_dim,
    embedding_dim,
    bias=False,
)

Q = W_q(X)
K = W_k(X)
V = W_v(X)

print("\nQ shape:", Q.shape)
print("K shape:", K.shape)
print("V shape:", V.shape)


# --------------------------------------------------
# ATTENTION SCORES
# --------------------------------------------------

scores = (
    Q @ K.transpose(-2, -1)
) / math.sqrt(embedding_dim)

print("\nRaw attention score shape:")
print(scores.shape)

print("\nRaw attention scores:")
print(scores)


# --------------------------------------------------
# CAUSAL MASK
# --------------------------------------------------

sequence_length = len(tokens)

causal_mask = torch.triu(
    torch.ones(
        sequence_length,
        sequence_length,
        dtype=torch.bool,
    ),
    diagonal=1,
)

scores = scores.masked_fill(
    causal_mask,
    float("-inf"),
)

print("\nMasked attention scores:")
print(scores)


# --------------------------------------------------
# SOFTMAX
# --------------------------------------------------

attention_weights = torch.softmax(
    scores,
    dim=-1,
)

print("\nAttention weights:")
print(attention_weights)


# --------------------------------------------------
# ATTENTION OUTPUT
# --------------------------------------------------

output = attention_weights @ V

print("\nAttention output shape:")
print(output.shape)

print("\nAttention output:")
print(output)


# --------------------------------------------------
# FRIENDLY DISPLAY
# --------------------------------------------------

print("\nAttention by token:\n")

for i, token in enumerate(tokens):

    print(f"{token:>5} attends to:")

    for j, source_token in enumerate(tokens):

        weight = attention_weights[i, j].item()

        print(
            f"    {source_token:>5}: "
            f"{weight:.4f}"
        )

    print()