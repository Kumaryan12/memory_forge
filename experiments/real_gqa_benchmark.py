import statistics
import time
from pathlib import Path

import pandas as pd
import torch

from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
)


MODEL_NAME = "Qwen/Qwen2.5-0.5B"

PROMPT_LENGTHS = [
    16,
    64,
    128,
    256,
    512,
]

NEW_TOKENS = 16
REPEATS = 3


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
# CACHE MEMORY
# ============================================================

def cache_size_bytes(cache):

    if cache is None:
        return 0

    total = 0


    # Modern Hugging Face Cache object
    if (
        hasattr(cache, "key_cache")
        and hasattr(cache, "value_cache")
    ):

        for K, V in zip(
            cache.key_cache,
            cache.value_cache
        ):

            total += (
                K.numel()
                * K.element_size()
            )

            total += (
                V.numel()
                * V.element_size()
            )

        return total


    # Legacy cache
    for layer_cache in cache:

        K = layer_cache[0]
        V = layer_cache[1]

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
# LOAD CONFIG FIRST
# ============================================================

config = AutoConfig.from_pretrained(
    MODEL_NAME
)


num_layers = config.num_hidden_layers

num_query_heads = (
    config.num_attention_heads
)

num_kv_heads = getattr(
    config,
    "num_key_value_heads",
    num_query_heads
)

hidden_size = (
    config.hidden_size
)

head_dim = getattr(
    config,
    "head_dim",
    hidden_size // num_query_heads
)


print("\nArchitecture")
print("-" * 50)

print(
    "Layers:",
    num_layers
)

print(
    "Hidden size:",
    hidden_size
)

print(
    "Query heads:",
    num_query_heads
)

print(
    "KV heads:",
    num_kv_heads
)

print(
    "Head dimension:",
    head_dim
)


if num_kv_heads == num_query_heads:

    attention_type = "MHA"

elif num_kv_heads == 1:

    attention_type = "MQA"

else:

    attention_type = "GQA"


print(
    "Attention type:",
    attention_type
)


# ============================================================
# MODEL
# ============================================================

print(
    "\nLoading model..."
)


tokenizer = (
    AutoTokenizer
    .from_pretrained(
        MODEL_NAME
    )
)


model = (
    AutoModelForCausalLM
    .from_pretrained(
        MODEL_NAME
    )
    .to(device)
)


model.eval()


parameter_count = sum(
    p.numel()
    for p in model.parameters()
)


print(
    "Parameters:",
    f"{parameter_count:,}"
)


# ============================================================
# CONTROLLED PROMPT
# ============================================================

def make_prompt(length):

    token_id = tokenizer.encode(
        " hello",
        add_special_tokens=False
    )[0]


    return torch.full(
        (1, length),
        token_id,
        dtype=torch.long,
        device=device
    )


# ============================================================
# GENERATION
# ============================================================

@torch.inference_mode()
def generate_no_cache(
    prompt,
    new_tokens
):

    sequence = prompt.clone()


    for _ in range(new_tokens):

        outputs = model(
            input_ids=sequence,
            use_cache=False
        )


        next_token = torch.argmax(
            outputs.logits[:, -1, :],
            dim=-1,
            keepdim=True
        )


        sequence = torch.cat(
            [sequence, next_token],
            dim=1
        )


    return sequence


@torch.inference_mode()
def generate_cached(
    prompt,
    new_tokens
):

    sequence = prompt.clone()


    synchronize()

    start = time.perf_counter()


    outputs = model(
        input_ids=prompt,
        use_cache=True
    )


    next_token = torch.argmax(
        outputs.logits[:, -1, :],
        dim=-1,
        keepdim=True
    )


    synchronize()

    ttft = (
        time.perf_counter()
        - start
    )


    cache = outputs.past_key_values


    sequence = torch.cat(
        [sequence, next_token],
        dim=1
    )


    synchronize()

    start = time.perf_counter()


    for _ in range(
        new_tokens - 1
    ):

        outputs = model(
            input_ids=next_token,
            past_key_values=cache,
            use_cache=True
        )


        cache = outputs.past_key_values


        next_token = torch.argmax(
            outputs.logits[:, -1, :],
            dim=-1,
            keepdim=True
        )


        sequence = torch.cat(
            [sequence, next_token],
            dim=1
        )


    synchronize()


    decode_time = (
        time.perf_counter()
        - start
    )


    return (
        sequence,
        ttft,
        decode_time,
        cache
    )


# ============================================================
# BENCHMARK
# ============================================================

rows = []


for prompt_length in PROMPT_LENGTHS:

    prompt = make_prompt(
        prompt_length
    )


    # warmup
    _ = generate_no_cache(
        prompt,
        2
    )

    _ = generate_cached(
        prompt,
        2
    )


    naive_times = []

    cached_times = []

    ttft_times = []

    decode_times = []

    cache_sizes = []


    for _ in range(REPEATS):

        synchronize()

        start = time.perf_counter()


        naive_output = (
            generate_no_cache(
                prompt,
                NEW_TOKENS
            )
        )


        synchronize()


        naive_time = (
            time.perf_counter()
            - start
        )


        naive_times.append(
            naive_time
        )


        (
            cached_output,
            ttft,
            decode_time,
            cache
        ) = generate_cached(
            prompt,
            NEW_TOKENS
        )


        ttft_times.append(
            ttft
        )

        decode_times.append(
            decode_time
        )

        cached_times.append(
            ttft
            + decode_time
        )

        cache_sizes.append(
            cache_size_bytes(
                cache
            )
        )


    naive = statistics.median(
        naive_times
    )

    cached = statistics.median(
        cached_times
    )

    ttft = statistics.median(
        ttft_times
    )

    decode = statistics.median(
        decode_times
    )

    cache_bytes = (
        statistics.median(
            cache_sizes
        )
    )


    outputs_match = torch.equal(
        naive_output,
        cached_output
    )


    decode_steps = (
        NEW_TOKENS - 1
    )


    row = {

        "model":
            MODEL_NAME,

        "parameters":
            parameter_count,

        "attention_type":
            attention_type,

        "layers":
            num_layers,

        "query_heads":
            num_query_heads,

        "kv_heads":
            num_kv_heads,

        "head_dim":
            head_dim,

        "prompt_length":
            prompt_length,

        "no_cache_ms":
            naive * 1000,

        "cached_ms":
            cached * 1000,

        "speedup_x":
            naive / cached,

        "ttft_ms":
            ttft * 1000,

        "tpot_ms":
            (
                decode
                / decode_steps
                * 1000
            ),

        "kv_cache_mib":
            cache_bytes
            / (1024 ** 2),

        "outputs_match":
            outputs_match
    }


    rows.append(row)


    print(
        f"Prompt {prompt_length:4d} | "
        f"No cache {naive * 1000:8.2f} ms | "
        f"Cache {cached * 1000:8.2f} ms | "
        f"Speedup {naive / cached:5.2f}x | "
        f"KV {cache_bytes / (1024 ** 2):7.2f} MiB"
    )


df = pd.DataFrame(
    rows
)


path = Path(
    "benchmarks/"
    "real_gqa_results.csv"
)


df.to_csv(
    path,
    index=False
)


print("\n")
print(df.to_string(index=False))

print(
    "\nSaved:",
    path
)