import gc
import statistics
import time
from pathlib import Path

import pandas as pd
import torch

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_NAME = "Qwen/Qwen2.5-0.5B"

PROMPT_LENGTHS = [
    64,
    128,
    256,
    512,
]

NEW_TOKENS = 16

REPEATS = 3

ATTENTION_IMPLEMENTATION = "sdpa"


PRECISION_CONFIGS = [
    {
        "name": "FP32",
        "dtype": torch.float32,
    },
    {
        "name": "BF16",
        "dtype": torch.bfloat16,
    },
]


# ============================================================
# DEVICE
# ============================================================

if torch.cuda.is_available():

    device = torch.device("cuda")

elif torch.backends.mps.is_available():

    device = torch.device("mps")

else:

    device = torch.device("cpu")


print("=" * 75)
print("MEMORYFORGE — PRECISION PERFORMANCE BENCHMARK")
print("=" * 75)

print("Device:", device)
print("Model :", MODEL_NAME)


# ============================================================
# SYNCHRONIZATION
# ============================================================

def synchronize():

    if device.type == "cuda":

        torch.cuda.synchronize()

    elif device.type == "mps":

        torch.mps.synchronize()


# ============================================================
# TOKENIZER
# ============================================================

print("\nLoading tokenizer...")


tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME
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
        (
            1,
            length
        ),
        token_id,
        dtype=torch.long,
        device=device
    )


# ============================================================
# CACHE MEMORY
# ============================================================

def cache_size_bytes(cache):

    if cache is None:

        return 0


    total = 0


    # --------------------------------------------------------
    # New Hugging Face DynamicCache API
    # --------------------------------------------------------

    if hasattr(cache, "layers"):

        for layer in cache.layers:

            K = layer.keys
            V = layer.values


            if K is not None:

                total += (
                    K.numel()
                    * K.element_size()
                )


            if V is not None:

                total += (
                    V.numel()
                    * V.element_size()
                )


        return total


    # --------------------------------------------------------
    # Older DynamicCache API
    # --------------------------------------------------------

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


    # --------------------------------------------------------
    # Legacy tuple cache
    # --------------------------------------------------------

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
# PARAMETER MEMORY
# ============================================================

def parameter_memory_bytes(model):

    total = 0


    for parameter in model.parameters():

        total += (
            parameter.numel()
            * parameter.element_size()
        )


    return total


# ============================================================
# BUFFER MEMORY
# ============================================================

def buffer_memory_bytes(model):

    total = 0


    for buffer in model.buffers():

        total += (
            buffer.numel()
            * buffer.element_size()
        )


    return total


# ============================================================
# OPTIONAL DEVICE ALLOCATOR MEMORY
# ============================================================

def current_device_allocated_memory():

    if device.type == "cuda":

        return (
            torch.cuda.memory_allocated()
        )


    if device.type == "mps":

        try:

            return (
                torch.mps.current_allocated_memory()
            )

        except Exception:

            return None


    return None


# ============================================================
# CACHED GENERATION
# ============================================================

@torch.inference_mode()
def generate_cached(
    model,
    prompt,
    new_tokens
):

    sequence = prompt.clone()


    # ========================================================
    # PREFILL / FIRST TOKEN
    # ========================================================

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


    cache = (
        outputs.past_key_values
    )


    sequence = torch.cat(
        [
            sequence,
            next_token
        ],
        dim=1
    )


    # ========================================================
    # DECODE
    # ========================================================

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


        cache = (
            outputs.past_key_values
        )


        next_token = torch.argmax(
            outputs.logits[:, -1, :],
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
# BENCHMARK ONE PRECISION
# ============================================================

def benchmark_precision(
    precision_name,
    dtype
):

    print(
        "\n\n"
        + "#" * 75
    )

    print(
        f"PRECISION: {precision_name}"
    )

    print(
        "#" * 75
    )


    # ========================================================
    # CLEAN PREVIOUS MEMORY
    # ========================================================

    gc.collect()


    if device.type == "mps":

        torch.mps.empty_cache()


    elif device.type == "cuda":

        torch.cuda.empty_cache()


    allocator_before = (
        current_device_allocated_memory()
    )


    # ========================================================
    # LOAD MODEL
    # ========================================================

    print(
        "Loading model..."
    )


    model = (
        AutoModelForCausalLM
        .from_pretrained(
            MODEL_NAME,
            dtype=dtype,
            attn_implementation=(
                ATTENTION_IMPLEMENTATION
            ),
        )
        .to(device)
    )


    model.eval()


    actual_dtype = next(
        model.parameters()
    ).dtype


    parameter_count = sum(
        p.numel()
        for p in model.parameters()
    )


    parameter_bytes = (
        parameter_memory_bytes(
            model
        )
    )


    buffer_bytes = (
        buffer_memory_bytes(
            model
        )
    )


    allocator_after_model = (
        current_device_allocated_memory()
    )


    print(
        "Parameters:",
        f"{parameter_count:,}"
    )

    print(
        "Requested dtype:",
        dtype
    )

    print(
        "Actual dtype:",
        actual_dtype
    )

    print(
        "Parameter storage:",
        f"{parameter_bytes / (1024 ** 2):.2f} MiB"
    )

    print(
        "Buffer storage:",
        f"{buffer_bytes / (1024 ** 2):.2f} MiB"
    )


    if (
        allocator_before is not None
        and allocator_after_model
        is not None
    ):

        allocator_delta = (
            allocator_after_model
            - allocator_before
        )


        print(
            "Device allocator increase:",
            f"{allocator_delta / (1024 ** 2):.2f} MiB"
        )

    else:

        allocator_delta = None


    # ========================================================
    # BENCHMARK EACH CONTEXT
    # ========================================================

    rows = []


    for prompt_length in (
        PROMPT_LENGTHS
    ):

        prompt = make_prompt(
            prompt_length
        )


        # ====================================================
        # WARMUP
        # ====================================================

        _ = generate_cached(
            model,
            prompt,
            2
        )


        ttft_times = []

        decode_times = []

        total_times = []

        cache_sizes = []

        generated_sequence = None


        # ====================================================
        # REPEATS
        # ====================================================

        for _ in range(REPEATS):

            (
                generated_sequence,
                ttft,
                decode_time,
                cache
            ) = generate_cached(
                model,
                prompt,
                NEW_TOKENS
            )


            ttft_times.append(
                ttft
            )


            decode_times.append(
                decode_time
            )


            total_times.append(
                ttft
                + decode_time
            )


            cache_sizes.append(
                cache_size_bytes(
                    cache
                )
            )


        # ====================================================
        # MEDIANS
        # ====================================================

        ttft = statistics.median(
            ttft_times
        )


        decode_time = statistics.median(
            decode_times
        )


        total_time = statistics.median(
            total_times
        )


        cache_bytes = statistics.median(
            cache_sizes
        )


        decode_steps = (
            NEW_TOKENS - 1
        )


        tpot_ms = (
            decode_time
            / decode_steps
            * 1000
        )


        decode_tokens_per_sec = (
            decode_steps
            / decode_time
        )


        end_to_end_tokens_per_sec = (
            NEW_TOKENS
            / total_time
        )


        final_cached_tokens = (
            prompt_length
            + NEW_TOKENS
            - 1
        )


        cache_bytes_per_token = (
            cache_bytes
            / final_cached_tokens
        )


        allocator_current = (
            current_device_allocated_memory()
        )


        row = {

            "precision":
                precision_name,

            "requested_dtype":
                str(dtype),

            "actual_dtype":
                str(actual_dtype),

            "attention_backend":
                ATTENTION_IMPLEMENTATION,

            "device":
                str(device),

            "parameters":
                parameter_count,

            "parameter_memory_mib":
                parameter_bytes
                / (1024 ** 2),

            "buffer_memory_mib":
                buffer_bytes
                / (1024 ** 2),

            "prompt_length":
                prompt_length,

            "new_tokens":
                NEW_TOKENS,

            "model_ttft_ms":
                ttft
                * 1000,

            "decode_ms":
                decode_time
                * 1000,

            "tpot_ms":
                tpot_ms,

            "cached_total_ms":
                total_time
                * 1000,

            "decode_tokens_per_sec":
                decode_tokens_per_sec,

            "end_to_end_tokens_per_sec":
                end_to_end_tokens_per_sec,

            "kv_cache_mib":
                cache_bytes
                / (1024 ** 2),

            "kv_bytes_per_cached_token":
                cache_bytes_per_token,

            "device_allocated_mib":
                (
                    allocator_current
                    / (1024 ** 2)
                    if allocator_current
                    is not None
                    else None
                ),
        }


        rows.append(
            row
        )


        print(
            f"Prompt {prompt_length:4d} | "
            f"TTFT {ttft * 1000:8.2f} ms | "
            f"TPOT {tpot_ms:7.3f} ms | "
            f"Decode {decode_tokens_per_sec:7.2f} tok/s | "
            f"KV {cache_bytes / (1024 ** 2):7.3f} MiB"
        )


    # ========================================================
    # FREE MODEL
    # ========================================================

    del model

    gc.collect()


    if device.type == "mps":

        torch.mps.empty_cache()


    elif device.type == "cuda":

        torch.cuda.empty_cache()


    return rows


# ============================================================
# RUN
# ============================================================

all_rows = []


for precision in PRECISION_CONFIGS:

    rows = benchmark_precision(

        precision_name=(
            precision["name"]
        ),

        dtype=(
            precision["dtype"]
        ),
    )


    all_rows.extend(
        rows
    )


# ============================================================
# SAVE
# ============================================================

df = pd.DataFrame(
    all_rows
)


output_path = Path(
    "benchmarks/"
    "precision_performance_results.csv"
)


output_path.parent.mkdir(
    parents=True,
    exist_ok=True
)


df.to_csv(
    output_path,
    index=False
)


# ============================================================
# SUMMARY
# ============================================================

print(
    "\n\n"
    + "=" * 100
)

print(
    "FINAL PRECISION RESULTS"
)

print(
    "=" * 100
)


columns = [
    "precision",
    "prompt_length",
    "parameter_memory_mib",
    "model_ttft_ms",
    "tpot_ms",
    "decode_tokens_per_sec",
    "kv_cache_mib",
]


print(
    df[
        columns
    ].to_string(
        index=False
    )
)


print(
    "\nSaved:",
    output_path
)