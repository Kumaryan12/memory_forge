import math

import torch
import torch.nn as nn


torch.manual_seed(42)

torch.set_printoptions(
    precision=3,
    sci_mode=False
)


# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

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


vocab_size = len(tokens)

embedding_dim = 8

num_heads = 2


assert embedding_dim % num_heads == 0


head_dim = embedding_dim // num_heads


print("Embedding dimension:", embedding_dim)
print("Number of heads:", num_heads)
print("Dimension per head:", head_dim)


# --------------------------------------------------
# EMBEDDINGS
# --------------------------------------------------

embedding = nn.Embedding(
    vocab_size,
    embedding_dim
)


X = embedding(token_ids)


print("\nInput X shape:")
print(X.shape)


# --------------------------------------------------
# Q K V PROJECTIONS
# --------------------------------------------------

W_q = nn.Linear(
    embedding_dim,
    embedding_dim,
    bias=False
)

W_k = nn.Linear(
    embedding_dim,
    embedding_dim,
    bias=False
)

W_v = nn.Linear(
    embedding_dim,
    embedding_dim,
    bias=False
)


Q = W_q(X)

K = W_k(X)

V = W_v(X)


print("\nOriginal shapes:")

print("Q:", Q.shape)
print("K:", K.shape)
print("V:", V.shape)


# --------------------------------------------------
# SPLIT INTO HEADS
# --------------------------------------------------

sequence_length = X.shape[0]


Q = Q.view(
    sequence_length,
    num_heads,
    head_dim
)

K = K.view(
    sequence_length,
    num_heads,
    head_dim
)

V = V.view(
    sequence_length,
    num_heads,
    head_dim
)


print("\nAfter splitting:")

print("Q:", Q.shape)
print("K:", K.shape)
print("V:", V.shape)


# --------------------------------------------------
# MOVE HEAD DIMENSION FIRST
# --------------------------------------------------

Q = Q.transpose(0, 1)

K = K.transpose(0, 1)

V = V.transpose(0, 1)


print("\nAfter transpose:")

print("Q:", Q.shape)
print("K:", K.shape)
print("V:", V.shape)


# --------------------------------------------------
# ATTENTION SCORES
# --------------------------------------------------

scores = (
    Q @ K.transpose(-2, -1)
) / math.sqrt(head_dim)


print("\nAttention score shape:")
print(scores.shape)


# --------------------------------------------------
# CAUSAL MASK
# --------------------------------------------------

mask = torch.triu(
    torch.ones(
        sequence_length,
        sequence_length,
        dtype=torch.bool
    ),
    diagonal=1
)


scores = scores.masked_fill(
    mask,
    float("-inf")
)


# --------------------------------------------------
# SOFTMAX
# --------------------------------------------------

attention_weights = torch.softmax(
    scores,
    dim=-1
)


print("\nAttention weight shape:")
print(attention_weights.shape)


# --------------------------------------------------
# ATTENTION OUTPUT PER HEAD
# --------------------------------------------------

head_output = (
    attention_weights @ V
)


print("\nOutput per head:")
print(head_output.shape)


# --------------------------------------------------
# CONCATENATE HEADS
# --------------------------------------------------

head_output = head_output.transpose(
    0,
    1
)


concatenated = head_output.reshape(
    sequence_length,
    embedding_dim
)


print("\nConcatenated shape:")
print(concatenated.shape)


# --------------------------------------------------
# OUTPUT PROJECTION
# --------------------------------------------------

W_o = nn.Linear(
    embedding_dim,
    embedding_dim,
    bias=False
)


output = W_o(concatenated)


print("\nFinal output shape:")
print(output.shape)


# --------------------------------------------------
# DISPLAY EACH HEAD
# --------------------------------------------------

for head in range(num_heads):

    print("\n" + "=" * 50)

    print(
        f"ATTENTION HEAD {head + 1}"
    )

    print("=" * 50)

    for i, query_token in enumerate(tokens):

        print(
            f"\n'{query_token}' attends to:"
        )

        for j, key_token in enumerate(tokens):

            weight = (
                attention_weights[
                    head,
                    i,
                    j
                ]
                .item()
            )

            print(
                f"  {key_token:<5}: "
                f"{weight:.3f}"
            )