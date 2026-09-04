import math

import torch
import torch.nn as nn


torch.manual_seed(42)

torch.set_printoptions(
    precision=3,
    sci_mode=False
)


# ==================================================
# CAUSAL MULTI-HEAD SELF-ATTENTION
# ==================================================

class CausalSelfAttention(nn.Module):

    def __init__(
        self,
        embedding_dim,
        num_heads
    ):
        super().__init__()

        assert embedding_dim % num_heads == 0

        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.head_dim = embedding_dim // num_heads


        # Q, K, V projections
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


        # Final output projection
        self.out_proj = nn.Linear(
            embedding_dim,
            embedding_dim,
            bias=False
        )


    def forward(self, x):

        # x shape:
        # [B, T, C]

        B, T, C = x.shape


        # ------------------------------------------
        # Q K V
        # ------------------------------------------

        Q = self.q_proj(x)
        K = self.k_proj(x)
        V = self.v_proj(x)


        # Current shape:
        #
        # [B, T, C]
        #
        # Convert to:
        #
        # [B, T, heads, head_dim]

        Q = Q.view(
            B,
            T,
            self.num_heads,
            self.head_dim
        )

        K = K.view(
            B,
            T,
            self.num_heads,
            self.head_dim
        )

        V = V.view(
            B,
            T,
            self.num_heads,
            self.head_dim
        )


        # Move heads before tokens
        #
        # [B, heads, T, head_dim]

        Q = Q.transpose(1, 2)
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)


        # ------------------------------------------
        # ATTENTION SCORES
        # ------------------------------------------

        scores = (
            Q @ K.transpose(-2, -1)
        ) / math.sqrt(self.head_dim)


        # scores:
        #
        # [B, heads, T, T]


        # ------------------------------------------
        # CAUSAL MASK
        # ------------------------------------------

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


        # ------------------------------------------
        # SOFTMAX
        # ------------------------------------------

        attention_weights = torch.softmax(
            scores,
            dim=-1
        )


        # ------------------------------------------
        # WEIGHTED VALUES
        # ------------------------------------------

        output = attention_weights @ V


        # output:
        #
        # [B, heads, T, head_dim]


        # ------------------------------------------
        # CONCATENATE HEADS
        # ------------------------------------------

        output = output.transpose(
            1,
            2
        )


        # [B, T, heads, head_dim]

        output = output.contiguous().view(
            B,
            T,
            C
        )


        # ------------------------------------------
        # OUTPUT PROJECTION
        # ------------------------------------------

        output = self.out_proj(output)


        return output, attention_weights


# ==================================================
# FEED-FORWARD NETWORK
# ==================================================

class FeedForward(nn.Module):

    def __init__(
        self,
        embedding_dim
    ):
        super().__init__()


        hidden_dim = 4 * embedding_dim


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


        self.attention = CausalSelfAttention(
            embedding_dim,
            num_heads
        )


        self.norm2 = nn.LayerNorm(
            embedding_dim
        )


        self.feed_forward = FeedForward(
            embedding_dim
        )


    def forward(self, x):

        # ------------------------------------------
        # ATTENTION + RESIDUAL
        # ------------------------------------------

        attention_input = self.norm1(x)


        attention_output, weights = (
            self.attention(
                attention_input
            )
        )


        x = x + attention_output


        # ------------------------------------------
        # FFN + RESIDUAL
        # ------------------------------------------

        ff_input = self.norm2(x)


        ff_output = self.feed_forward(
            ff_input
        )


        x = x + ff_output


        return x, weights


# ==================================================
# TOY INPUT
# ==================================================

tokens = [
    "the",
    "cat",
    "sat",
    "down"
]


token_ids = torch.tensor([
    [0, 1, 2, 3]
])


vocab_size = len(tokens)

embedding_dim = 8

num_heads = 2


# ==================================================
# TOKEN EMBEDDINGS
# ==================================================

embedding = nn.Embedding(
    vocab_size,
    embedding_dim
)


X = embedding(
    token_ids
)


print("\nInput shape:")
print(X.shape)


# ==================================================
# TRANSFORMER BLOCK
# ==================================================

block = TransformerBlock(
    embedding_dim=embedding_dim,
    num_heads=num_heads
)


output, attention_weights = block(X)


print("\nTransformer output shape:")
print(output.shape)


print("\nTransformer output:")
print(output)


print("\nAttention shape:")
print(attention_weights.shape)


# ==================================================
# DISPLAY ATTENTION
# ==================================================

for head in range(num_heads):

    print("\n" + "=" * 50)

    print(
        f"HEAD {head + 1}"
    )

    print("=" * 50)


    for i, query_token in enumerate(tokens):

        print(
            f"\n'{query_token}' attends to:"
        )


        for j, key_token in enumerate(tokens):

            weight = (
                attention_weights[
                    0,
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