import math
import statistics
import time
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn


# ============================================================
# REPRODUCIBILITY
# ============================================================

torch.manual_seed(42)


# ============================================================
# DEVICE
# ============================================================

if torch.cuda.is_available():

    device = torch.device("cuda")

elif torch.backends.mps.is_available():

    device = torch.device("mps")

else:

    device = torch.device("cpu")


print("Device:", device)


def synchronize():

    if device.type == "cuda":

        torch.cuda.synchronize()

    elif device.type == "mps":

        torch.mps.synchronize()


# ============================================================
# GROUPED-QUERY ATTENTION
# ============================================================

class GroupedQueryAttention(nn.Module):

    def __init__(
        self,
        embedding_dim,
        num_query_heads,
        num_kv_heads
    ):
        super().__init__()

        assert (
            embedding_dim
            % num_query_heads
            == 0
        )

        assert (
            num_query_heads
            % num_kv_heads
            == 0
        )


        self.embedding_dim = (
            embedding_dim
        )

        self.num_query_heads = (
            num_query_heads
        )

        self.num_kv_heads = (
            num_kv_heads
        )

        self.head_dim = (
            embedding_dim
            // num_query_heads
        )

        self.group_size = (
            num_query_heads
            // num_kv_heads
        )


        # ----------------------------------------------------
        # QUERY PROJECTION
        #
        # still produces all Query heads
        # ----------------------------------------------------

        self.q_proj = nn.Linear(
            embedding_dim,
            num_query_heads
            * self.head_dim,
            bias=False
        )


        # ----------------------------------------------------
        # KEY / VALUE PROJECTIONS
        #
        # output becomes smaller as H_KV decreases
        # ----------------------------------------------------

        kv_dim = (
            num_kv_heads
            * self.head_dim
        )


        self.k_proj = nn.Linear(
            embedding_dim,
            kv_dim,
            bias=False
        )


        self.v_proj = nn.Linear(
            embedding_dim,
            kv_dim,
            bias=False
        )


        self.out_proj = nn.Linear(
            embedding_dim,
            embedding_dim,
            bias=False
        )


    def forward(
        self,
        x,
        past_kv=None,
        use_cache=False
    ):

        B, T, C = x.shape


        # ====================================================
        # QUERY
        # ====================================================

        Q = self.q_proj(x)


        Q = Q.view(
            B,
            T,
            self.num_query_heads,
            self.head_dim
        ).transpose(
            1,
            2
        )


        # Q:
        #
        # [B, Hq, T, Dh]


        # ====================================================
        # KEY / VALUE
        # ====================================================

        K_new = self.k_proj(x)

        V_new = self.v_proj(x)


        K_new = K_new.view(
            B,
            T,
            self.num_kv_heads,
            self.head_dim
        ).transpose(
            1,
            2
        )


        V_new = V_new.view(
            B,
            T,
            self.num_kv_heads,
            self.head_dim
        ).transpose(
            1,
            2
        )


        # K/V:
        #
        # [B, Hkv, T, Dh]


        # ====================================================
        # CACHE
        # ====================================================

        if past_kv is not None:

            past_K, past_V = past_kv


            K = torch.cat(
                [past_K, K_new],
                dim=2
            )


            V = torch.cat(
                [past_V, V_new],
                dim=2
            )


            past_length = (
                past_K.size(2)
            )

        else:

            K = K_new

            V = V_new

            past_length = 0


        total_key_length = (
            K.size(2)
        )


        # ====================================================
        # GROUP QUERY HEADS
        # ====================================================
        #
        # Example:
        #
        # Hq  = 8
        # Hkv = 2
        #
        # group_size = 4
        #
        # Q shape becomes conceptually:
        #
        # [B, 2 KV groups, 4 Q heads/group, T, Dh]
        # ====================================================

        Q_grouped = Q.view(
            B,
            self.num_kv_heads,
            self.group_size,
            T,
            self.head_dim
        )


        # ====================================================
        # ATTENTION SCORES
        # ============================================================
        #
        # Q:
        # [B, Hkv, group, Tq, Dh]
        #
        # K:
        # [B, Hkv, Tk, Dh]
        #
        # result:
        #
        # [B, Hkv, group, Tq, Tk]
        # ====================================================

        scores = torch.einsum(
            "bhgtd,bhkd->bhgtk",
            Q_grouped,
            K
        )


        scores = (
            scores
            / math.sqrt(
                self.head_dim
            )
        )


        # ====================================================
        # CAUSAL MASK
        # ====================================================

        query_positions = (
            past_length
            + torch.arange(
                T,
                device=x.device
            )
        )


        key_positions = torch.arange(
            total_key_length,
            device=x.device
        )


        mask = (
            key_positions.unsqueeze(0)
            >
            query_positions.unsqueeze(1)
        )


        mask = mask.view(
            1,
            1,
            1,
            T,
            total_key_length
        )


        scores = scores.masked_fill(
            mask,
            float("-inf")
        )


        # ====================================================
        # SOFTMAX
        # ====================================================

        weights = torch.softmax(
            scores,
            dim=-1
        )


        # ====================================================
        # RETRIEVE VALUES
        # ====================================================

        output = torch.einsum(
            "bhgtk,bhkd->bhgtd",
            weights,
            V
        )


        # ----------------------------------------------------
        # Merge groups back into Query heads
        # ----------------------------------------------------

        output = output.reshape(
            B,
            self.num_query_heads,
            T,
            self.head_dim
        )


        output = (
            output
            .transpose(1, 2)
            .contiguous()
            .view(B, T, C)
        )


        output = self.out_proj(
            output
        )


        present_kv = None


        if use_cache:

            # Important:
            #
            # We store the COMPACT K/V representation.
            #
            # We do not duplicate KV heads
            # to match Query heads.

            present_kv = (
                K,
                V
            )


        return (
            output,
            present_kv
        )


# ============================================================
# FEED-FORWARD
# ============================================================

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


# ============================================================
# TRANSFORMER BLOCK
# ============================================================

class TransformerBlock(nn.Module):

    def __init__(
        self,
        embedding_dim,
        num_query_heads,
        num_kv_heads
    ):
        super().__init__()


        self.norm1 = nn.LayerNorm(
            embedding_dim
        )


        self.attention = (
            GroupedQueryAttention(
                embedding_dim,
                num_query_heads,
                num_kv_heads
            )
        )


        self.norm2 = nn.LayerNorm(
            embedding_dim
        )


        self.ffn = FeedForward(
            embedding_dim
        )


    def forward(
        self,
        x,
        past_kv=None,
        use_cache=False
    ):

        attention_output, cache = (
            self.attention(
                self.norm1(x),
                past_kv=past_kv,
                use_cache=use_cache
            )
        )


        x = (
            x
            + attention_output
        )


        x = (
            x
            + self.ffn(
                self.norm2(x)
            )
        )


        return (
            x,
            cache
        )


# ============================================================
# TINY DECODER MODEL
# ============================================================

class TinyDecoder(nn.Module):

    def __init__(
        self,
        vocab_size,
        max_seq_len,
        embedding_dim,
        num_query_heads,
        num_kv_heads,
        num_layers
    ):
        super().__init__()


        self.max_seq_len = (
            max_seq_len
        )

        self.embedding_dim = (
            embedding_dim
        )

        self.num_query_heads = (
            num_query_heads
        )

        self.num_kv_heads = (
            num_kv_heads
        )

        self.num_layers = (
            num_layers
        )

        self.head_dim = (
            embedding_dim
            // num_query_heads
        )


        self.token_embedding = nn.Embedding(
            vocab_size,
            embedding_dim
        )


        self.position_embedding = nn.Embedding(
            max_seq_len,
            embedding_dim
        )


        self.blocks = nn.ModuleList([

            TransformerBlock(
                embedding_dim,
                num_query_heads,
                num_kv_heads
            )

            for _ in range(
                num_layers
            )
        ])


        self.final_norm = nn.LayerNorm(
            embedding_dim
        )


        self.lm_head = nn.Linear(
            embedding_dim,
            vocab_size,
            bias=False
        )


    def forward(
        self,
        idx,
        past_key_values=None,
        use_cache=False
    ):

        B, T = idx.shape


        if past_key_values is None:

            past_key_values = [
                None
                for _ in self.blocks
            ]

            past_length = 0

        else:

            first_cache = (
                past_key_values[0]
            )

            if first_cache is None:

                past_length = 0

            else:

                past_length = (
                    first_cache[0]
                    .size(2)
                )


        if (
            past_length + T
            > self.max_seq_len
        ):

            raise ValueError(
                "Sequence exceeds max_seq_len"
            )


        token_embeddings = (
            self.token_embedding(idx)
        )


        positions = torch.arange(
            past_length,
            past_length + T,
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


        new_cache = []


        for layer_index, block in enumerate(
            self.blocks
        ):

            x, layer_cache = block(
                x,
                past_kv=(
                    past_key_values[
                        layer_index
                    ]
                ),
                use_cache=use_cache
            )


            if use_cache:

                new_cache.append(
                    layer_cache
                )


        x = self.final_norm(x)


        logits = self.lm_head(x)


        if not use_cache:

            new_cache = None


        return (
            logits,
            new_cache
        )


# ============================================================
# CACHE MEMORY
# ============================================================

def cache_size_bytes(cache):

    total = 0


    for K, V in cache:

        total += (
            K.numel()
            * K.element_size()
        )

        total += (
            V.numel()
            * V.element_size()
        )


    return total


# ============================================================
# THEORETICAL CACHE
# ============================================================

def theoretical_cache_bytes(
    num_layers,
    num_kv_heads,
    head_dim,
    context_length,
    bytes_per_value,
    batch_size=1
):

    return (

        2

        * batch_size

        * num_layers

        * num_kv_heads

        * context_length

        * head_dim

        * bytes_per_value

    )


# ============================================================
# CONFIG
# ============================================================

VOCAB_SIZE = 1024

MAX_SEQ_LEN = 1024

EMBEDDING_DIM = 256

NUM_QUERY_HEADS = 8

NUM_LAYERS = 6


ATTENTION_CONFIGS = {

    "MHA":
        8,

    "GQA":
        2,

    "MQA":
        1,
}


CONTEXT_LENGTHS = [
    64,
    128,
    256,
    512
]


REPEATS = 5


# ============================================================
# BENCHMARK
# ============================================================

rows = []


for attention_type, num_kv_heads in (
    ATTENTION_CONFIGS.items()
):

    print(
        "\n"
        + "=" * 70
    )

    print(
        f"{attention_type}: "
        f"Hq={NUM_QUERY_HEADS}, "
        f"Hkv={num_kv_heads}"
    )

    print(
        "=" * 70
    )


    model = TinyDecoder(

        vocab_size=VOCAB_SIZE,

        max_seq_len=MAX_SEQ_LEN,

        embedding_dim=(
            EMBEDDING_DIM
        ),

        num_query_heads=(
            NUM_QUERY_HEADS
        ),

        num_kv_heads=(
            num_kv_heads
        ),

        num_layers=(
            NUM_LAYERS
        )

    ).to(device)


    model.eval()


    parameter_count = sum(
        p.numel()
        for p in model.parameters()
    )


    print(
        "Parameters:",
        f"{parameter_count:,}"
    )


    bytes_per_value = (
        next(
            model.parameters()
        ).element_size()
    )


    for context_length in (
        CONTEXT_LENGTHS
    ):

        prompt = torch.randint(

            0,
            VOCAB_SIZE,

            (
                1,
                context_length
            ),

            device=device
        )


        # ====================================================
        # WARMUP
        # ====================================================

        with torch.inference_mode():

            _ = model(
                prompt,
                use_cache=True
            )


        # ====================================================
        # PREFILL MEASUREMENT
        # ====================================================

        prefill_times = []

        cache = None

        logits = None


        for _ in range(REPEATS):

            synchronize()

            start = (
                time.perf_counter()
            )


            with torch.inference_mode():

                logits, cache = model(
                    prompt,
                    use_cache=True
                )


            synchronize()

            end = (
                time.perf_counter()
            )


            prefill_times.append(
                end - start
            )


        prefill_time = (
            statistics.median(
                prefill_times
            )
        )


        # ====================================================
        # SINGLE-TOKEN DECODE
        # ====================================================

        next_token = torch.argmax(
            logits[:, -1, :],
            dim=-1,
            keepdim=True
        )


        decode_times = []


        for _ in range(REPEATS):

            synchronize()

            start = (
                time.perf_counter()
            )


            with torch.inference_mode():

                _ = model(
                    next_token,
                    past_key_values=cache,
                    use_cache=True
                )


            synchronize()

            end = (
                time.perf_counter()
            )


            decode_times.append(
                end - start
            )


        decode_time = (
            statistics.median(
                decode_times
            )
        )


        # ====================================================
        # MEMORY
        # ====================================================

        actual_bytes = (
            cache_size_bytes(
                cache
            )
        )


        theoretical_bytes = (
            theoretical_cache_bytes(

                num_layers=(
                    NUM_LAYERS
                ),

                num_kv_heads=(
                    num_kv_heads
                ),

                head_dim=(
                    model.head_dim
                ),

                context_length=(
                    context_length
                ),

                bytes_per_value=(
                    bytes_per_value
                ),
            )
        )


        row = {

            "attention_type":
                attention_type,

            "query_heads":
                NUM_QUERY_HEADS,

            "kv_heads":
                num_kv_heads,

            "context_length":
                context_length,

            "parameters":
                parameter_count,

            "prefill_ms":
                prefill_time * 1000,

            "decode_step_ms":
                decode_time * 1000,

            "actual_cache_mib":
                actual_bytes
                / (1024 ** 2),

            "theoretical_cache_mib":
                theoretical_bytes
                / (1024 ** 2),
        }


        rows.append(row)


        print(
            f"Context {context_length:4d} | "
            f"Prefill {prefill_time * 1000:8.2f} ms | "
            f"Decode {decode_time * 1000:7.3f} ms | "
            f"KV {actual_bytes / (1024 ** 2):7.3f} MiB"
        )


    del model


    if device.type == "mps":

        torch.mps.empty_cache()


    elif device.type == "cuda":

        torch.cuda.empty_cache()


# ============================================================
# SAVE
# ============================================================

df = pd.DataFrame(
    rows
)


output_path = Path(
    "benchmarks/"
    "gqa_mqa_results.csv"
)


output_path.parent.mkdir(
    parents=True,
    exist_ok=True
)


df.to_csv(
    output_path,
    index=False
)


print("\n")
print(df.to_string(index=False))


print(
    "\nSaved:",
    output_path
)