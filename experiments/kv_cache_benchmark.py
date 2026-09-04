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


print(f"Using device: {device}")


def synchronize():
    """
    GPU operations are asynchronous.

    We synchronize before/after timing so that
    perf_counter measures the actual device work.
    """

    if device.type == "cuda":
        torch.cuda.synchronize()

    elif device.type == "mps":
        torch.mps.synchronize()


# ============================================================
# CACHE-AWARE SELF ATTENTION
# ============================================================

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

        self.head_dim = (
            embedding_dim // num_heads
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


    def forward(
        self,
        x,
        past_kv=None,
        use_cache=False
    ):

        B, T, C = x.shape


        # ----------------------------------------------------
        # CURRENT TOKEN(S) Q K V
        # ----------------------------------------------------

        Q = self.q_proj(x)
        K_new = self.k_proj(x)
        V_new = self.v_proj(x)


        # [B, T, C]
        #
        #       ↓
        #
        # [B, H, T, Dh]

        Q = Q.view(
            B,
            T,
            self.num_heads,
            self.head_dim
        ).transpose(1, 2)


        K_new = K_new.view(
            B,
            T,
            self.num_heads,
            self.head_dim
        ).transpose(1, 2)


        V_new = V_new.view(
            B,
            T,
            self.num_heads,
            self.head_dim
        ).transpose(1, 2)


        # ----------------------------------------------------
        # ADD PREVIOUS K/V CACHE
        # ----------------------------------------------------

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

            past_length = past_K.size(2)

        else:

            K = K_new
            V = V_new

            past_length = 0


        total_key_length = K.size(2)


        # ----------------------------------------------------
        # ATTENTION SCORES
        # ----------------------------------------------------

        scores = (
            Q @ K.transpose(-2, -1)
        ) / math.sqrt(
            self.head_dim
        )


        # scores:
        #
        # [B, H, query_length, key_length]


        # ----------------------------------------------------
        # GENERAL CAUSAL MASK
        # ----------------------------------------------------
        #
        # This works both for:
        #
        # full prompt:
        # T > 1
        #
        # and incremental decoding:
        # T = 1
        #

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


        causal_mask = (
            key_positions.unsqueeze(0)
            >
            query_positions.unsqueeze(1)
        )


        causal_mask = causal_mask.view(
            1,
            1,
            T,
            total_key_length
        )


        scores = scores.masked_fill(
            causal_mask,
            float("-inf")
        )


        # ----------------------------------------------------
        # SOFTMAX
        # ----------------------------------------------------

        attention_weights = torch.softmax(
            scores,
            dim=-1
        )


        # ----------------------------------------------------
        # RETRIEVE VALUES
        # ----------------------------------------------------

        output = (
            attention_weights @ V
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


        # ----------------------------------------------------
        # CACHE
        # ----------------------------------------------------

        present_kv = None

        if use_cache:

            present_kv = (
                K,
                V
            )


        return output, present_kv


# ============================================================
# FEED-FORWARD NETWORK
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


    def forward(
        self,
        x,
        past_kv=None,
        use_cache=False
    ):

        attention_output, present_kv = (
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
            present_kv
        )


# ============================================================
# CACHE-AWARE TINY GPT
# ============================================================

class TinyGPT(nn.Module):

    def __init__(
        self,
        vocab_size,
        max_seq_len,
        embedding_dim=128,
        num_heads=4,
        num_layers=4
    ):
        super().__init__()


        self.vocab_size = vocab_size

        self.max_seq_len = max_seq_len

        self.embedding_dim = embedding_dim

        self.num_heads = num_heads

        self.num_layers = num_layers

        self.head_dim = (
            embedding_dim // num_heads
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
                num_heads
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


        # ----------------------------------------------------
        # DETERMINE HOW MANY TOKENS ARE ALREADY CACHED
        # ----------------------------------------------------

        if past_key_values is None:

            past_key_values = [
                None
                for _ in self.blocks
            ]

            past_length = 0

        else:

            first_layer_cache = (
                past_key_values[0]
            )

            if first_layer_cache is None:

                past_length = 0

            else:

                past_length = (
                    first_layer_cache[0]
                    .size(2)
                )


        if (
            past_length + T
            > self.max_seq_len
        ):

            raise ValueError(
                "Sequence exceeds max_seq_len."
            )


        # ----------------------------------------------------
        # TOKEN EMBEDDING
        # ----------------------------------------------------

        token_embeddings = (
            self.token_embedding(idx)
        )


        # ----------------------------------------------------
        # POSITION EMBEDDING
        # ----------------------------------------------------
        #
        # During cached decoding:
        #
        # prompt may occupy positions 0..31
        #
        # new token must therefore use
        # position 32
        #

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


        # ----------------------------------------------------
        # TRANSFORMER BLOCKS
        # ----------------------------------------------------

        present_key_values = []


        for layer_index, block in enumerate(
            self.blocks
        ):

            x, present_kv = block(
                x,
                past_kv=(
                    past_key_values[
                        layer_index
                    ]
                ),
                use_cache=use_cache
            )


            if use_cache:

                present_key_values.append(
                    present_kv
                )


        x = self.final_norm(x)


        logits = self.lm_head(x)


        if not use_cache:

            present_key_values = None


        return (
            logits,
            present_key_values
        )


# ============================================================
# NAIVE GENERATION
# ============================================================

@torch.inference_mode()
def generate_naive(
    model,
    prompt,
    max_new_tokens
):

    sequence = prompt.clone()


    for _ in range(
        max_new_tokens
    ):

        logits, _ = model(
            sequence,
            use_cache=False
        )


        next_token = torch.argmax(
            logits[:, -1, :],
            dim=-1,
            keepdim=True
        )


        sequence = torch.cat(
            [
                sequence,
                next_token
            ],
            dim=1
        )


    return sequence


# ============================================================
# CACHED GENERATION
# ============================================================

@torch.inference_mode()
def generate_cached(
    model,
    prompt,
    max_new_tokens
):

    sequence = prompt.clone()


    if max_new_tokens == 0:

        return (
            sequence,
            0.0,
            0.0,
            None
        )


    # --------------------------------------------------------
    # PREFILL
    # --------------------------------------------------------

    synchronize()

    prefill_start = (
        time.perf_counter()
    )


    logits, cache = model(
        prompt,
        use_cache=True
    )


    synchronize()

    prefill_end = (
        time.perf_counter()
    )


    prefill_time = (
        prefill_end
        - prefill_start
    )


    # First generated token comes from
    # the prefill output.

    next_token = torch.argmax(
        logits[:, -1, :],
        dim=-1,
        keepdim=True
    )


    sequence = torch.cat(
        [
            sequence,
            next_token
        ],
        dim=1
    )


    # --------------------------------------------------------
    # DECODE
    # --------------------------------------------------------

    synchronize()

    decode_start = (
        time.perf_counter()
    )


    for _ in range(
        max_new_tokens - 1
    ):

        logits, cache = model(
            next_token,
            past_key_values=cache,
            use_cache=True
        )


        next_token = torch.argmax(
            logits[:, -1, :],
            dim=-1,
            keepdim=True
        )


        sequence = torch.cat(
            [
                sequence,
                next_token
            ],
            dim=1
        )


    synchronize()

    decode_end = (
        time.perf_counter()
    )


    decode_time = (
        decode_end
        - decode_start
    )


    return (
        sequence,
        prefill_time,
        decode_time,
        cache
    )


# ============================================================
# CACHE SIZE
# ============================================================

def cache_size_bytes(
    cache
):

    if cache is None:
        return 0


    total_bytes = 0


    for K, V in cache:

        total_bytes += (
            K.numel()
            * K.element_size()
        )

        total_bytes += (
            V.numel()
            * V.element_size()
        )


    return total_bytes


# ============================================================
# THEORETICAL FINAL CACHE
# ============================================================

def theoretical_cache_mib(
    model,
    batch_size,
    context_length
):

    bytes_per_value = (
        next(
            model.parameters()
        ).element_size()
    )


    total_bytes = (

        2

        * model.num_layers

        * batch_size

        * model.num_heads

        * context_length

        * model.head_dim

        * bytes_per_value

    )


    return (
        total_bytes
        / (1024 ** 2)
    )


# ============================================================
# MODEL
# ============================================================

VOCAB_SIZE = 256

MAX_SEQ_LEN = 512

EMBEDDING_DIM = 256

NUM_HEADS = 8

NUM_LAYERS = 6


model = TinyGPT(
    vocab_size=VOCAB_SIZE,
    max_seq_len=MAX_SEQ_LEN,
    embedding_dim=EMBEDDING_DIM,
    num_heads=NUM_HEADS,
    num_layers=NUM_LAYERS
).to(device)


model.eval()


parameter_count = sum(
    p.numel()
    for p in model.parameters()
)


print(
    f"Parameters: "
    f"{parameter_count:,}"
)


# ============================================================
# CORRECTNESS TEST
# ============================================================

print("\nChecking cache correctness...")


test_prompt = torch.randint(
    0,
    VOCAB_SIZE,
    (1, 16),
    device=device
)


naive_output = generate_naive(
    model,
    test_prompt,
    max_new_tokens=8
)


cached_output, _, _, _ = (
    generate_cached(
        model,
        test_prompt,
        max_new_tokens=8
    )
)


same_output = torch.equal(
    naive_output,
    cached_output
)


print(
    "Naive and cached generation match:",
    same_output
)


# ============================================================
# BENCHMARK
# ============================================================

PROMPT_LENGTHS = [
    16,
    32,
    64,
    128,
    256
]

NEW_TOKENS = 32

REPEATS = 7


results = []


print("\nStarting benchmark...\n")


for prompt_length in PROMPT_LENGTHS:

    prompt = torch.randint(
        0,
        VOCAB_SIZE,
        (1, prompt_length),
        device=device
    )


    # --------------------------------------------------------
    # WARM-UP
    # --------------------------------------------------------

    _ = generate_naive(
        model,
        prompt,
        max_new_tokens=2
    )

    _ = generate_cached(
        model,
        prompt,
        max_new_tokens=2
    )


    naive_times = []

    cached_prefill_times = []

    cached_decode_times = []

    cached_total_times = []


    # --------------------------------------------------------
    # REPEATED MEASUREMENTS
    # --------------------------------------------------------

    for _ in range(REPEATS):

        synchronize()

        start = time.perf_counter()


        naive_sequence = generate_naive(
            model,
            prompt,
            max_new_tokens=NEW_TOKENS
        )


        synchronize()

        end = time.perf_counter()


        naive_times.append(
            end - start
        )


        (
            cached_sequence,
            prefill_time,
            decode_time,
            cache
        ) = generate_cached(
            model,
            prompt,
            max_new_tokens=NEW_TOKENS
        )


        cached_prefill_times.append(
            prefill_time
        )

        cached_decode_times.append(
            decode_time
        )

        cached_total_times.append(
            prefill_time
            + decode_time
        )


    # --------------------------------------------------------
    # MEDIANS
    # --------------------------------------------------------

    naive_time = statistics.median(
        naive_times
    )

    cached_prefill = statistics.median(
        cached_prefill_times
    )

    cached_decode = statistics.median(
        cached_decode_times
    )

    cached_total = statistics.median(
        cached_total_times
    )


    speedup = (
        naive_time
        / cached_total
    )


    naive_tokens_per_sec = (
        NEW_TOKENS
        / naive_time
    )


    cached_tokens_per_sec = (
        NEW_TOKENS
        / cached_total
    )


    actual_cache_mib = (
        cache_size_bytes(cache)
        / (1024 ** 2)
    )


    cached_context_length = (
        prompt_length
        + NEW_TOKENS
        - 1
    )


    theoretical_mib = (
        theoretical_cache_mib(
            model,
            batch_size=1,
            context_length=(
                cached_context_length
            )
        )
    )


    row = {

        "device":
            str(device),

        "prompt_length":
            prompt_length,

        "new_tokens":
            NEW_TOKENS,

        "naive_ms":
            naive_time * 1000,

        "cached_prefill_ms":
            cached_prefill * 1000,

        "cached_decode_ms":
            cached_decode * 1000,

        "cached_total_ms":
            cached_total * 1000,

        "speedup_x":
            speedup,

        "naive_tokens_per_sec":
            naive_tokens_per_sec,

        "cached_tokens_per_sec":
            cached_tokens_per_sec,

        "actual_cache_mib":
            actual_cache_mib,

        "theoretical_final_cache_mib":
            theoretical_mib
    }


    results.append(row)


    print(
        f"Prompt {prompt_length:3d} | "
        f"Naive {naive_time * 1000:8.2f} ms | "
        f"Cached {cached_total * 1000:8.2f} ms | "
        f"Speedup {speedup:5.2f}x"
    )


# ============================================================
# SAVE RESULTS
# ============================================================

df = pd.DataFrame(
    results
)


output_path = Path(
    "benchmarks/"
    "kv_cache_results.csv"
)


output_path.parent.mkdir(
    parents=True,
    exist_ok=True
)


df.to_csv(
    output_path,
    index=False
)


print("\nBenchmark results:\n")


print(
    df.to_string(
        index=False
    )
)


print(
    "\nSaved to:",
    output_path
)