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

MODELS = [
    "distilgpt2",
    "gpt2",
]

PROMPT_LENGTHS = [
    16,
    32,
    64,
    128,
    256,
    512,
]

NEW_TOKENS = 32

REPEATS = 5

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

def cache_size_bytes(cache):
    """
    Calculate storage occupied by K/V tensors.

    Supports several Hugging Face cache representations.
    """

    if cache is None:
        return 0


    # --------------------------------------------------------
    # Modern DynamicCache-style objects
    # --------------------------------------------------------

    if (
        hasattr(cache, "key_cache")
        and hasattr(cache, "value_cache")
    ):

        total = 0

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
    # Legacy tuple:
    #
    # (
    #   (K_layer0, V_layer0),
    #   (K_layer1, V_layer1),
    #   ...
    # )
    # --------------------------------------------------------

    total = 0

    try:

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

    except Exception as exc:

        print(
            "Warning: could not measure cache:",
            exc
        )

        return 0

def make_prompt(
    tokenizer,
    length
):

    token_ids = tokenizer.encode(
        " hello",
        add_special_tokens=False
    )

    token_id = token_ids[0]

    return torch.full(
        (1, length),
        token_id,
        dtype=torch.long,
        device=device
    )

@torch.inference_mode()
def generate_no_cache(
    model,
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
            [
                sequence,
                next_token
            ],
            dim=1
        )


    return sequence

@torch.inference_mode()
def generate_with_cache(
    model,
    prompt,
    new_tokens
):

    sequence = prompt.clone()


    # ========================================================
    # PREFILL / FIRST TOKEN
    # ========================================================

    synchronize()

    ttft_start = time.perf_counter()


    outputs = model(
        input_ids=prompt,
        use_cache=True
    )


    cache = outputs.past_key_values


    next_token = torch.argmax(
        outputs.logits[:, -1, :],
        dim=-1,
        keepdim=True
    )


    synchronize()

    ttft_end = time.perf_counter()


    model_ttft = (
        ttft_end
        - ttft_start
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

    decode_start = time.perf_counter()


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
            [
                sequence,
                next_token
            ],
            dim=1
        )


    synchronize()

    decode_end = time.perf_counter()


    decode_time = (
        decode_end
        - decode_start
    )


    return (
        sequence,
        model_ttft,
        decode_time,
        cache
    )

def benchmark_model(
    model_name
):

    print(
        "\n"
        + "=" * 70
    )

    print(
        f"MODEL: {model_name}"
    )

    print(
        "=" * 70
    )


    tokenizer = (
        AutoTokenizer
        .from_pretrained(
            model_name
        )
    )


    model = (
        AutoModelForCausalLM
        .from_pretrained(
            model_name
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


    rows = []


    for prompt_length in PROMPT_LENGTHS:

        prompt = make_prompt(
            tokenizer,
            prompt_length
        )


        # ====================================================
        # WARM-UP
        # ====================================================

        _ = generate_no_cache(
            model,
            prompt,
            2
        )


        _ = generate_with_cache(
            model,
            prompt,
            2
        )


        naive_times = []

        ttft_times = []

        decode_times = []

        cached_total_times = []

        cache_sizes = []


        naive_output = None
        cached_output = None


        # ====================================================
        # REPEATED RUNS
        # ====================================================

        for _ in range(REPEATS):

            # ------------------------------------------------
            # NO CACHE
            # ------------------------------------------------

            synchronize()

            start = time.perf_counter()


            naive_output = generate_no_cache(
                model,
                prompt,
                NEW_TOKENS
            )


            synchronize()

            end = time.perf_counter()


            naive_times.append(
                end - start
            )


            # ------------------------------------------------
            # CACHE
            # ------------------------------------------------

            (
                cached_output,
                model_ttft,
                decode_time,
                cache
            ) = generate_with_cache(
                model,
                prompt,
                NEW_TOKENS
            )


            ttft_times.append(
                model_ttft
            )

            decode_times.append(
                decode_time
            )

            cached_total_times.append(
                model_ttft
                + decode_time
            )


            cache_sizes.append(
                cache_size_bytes(
                    cache
                )
            )


        # ====================================================
        # MEDIAN METRICS
        # ====================================================

        naive_time = statistics.median(
            naive_times
        )


        model_ttft = statistics.median(
            ttft_times
        )


        decode_time = statistics.median(
            decode_times
        )


        cached_total = statistics.median(
            cached_total_times
        )


        cache_bytes = statistics.median(
            cache_sizes
        )


        speedup = (
            naive_time
            / cached_total
        )


        # First output token came from prefill.
        #
        # The decode loop processed:
        #
        # NEW_TOKENS - 1

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


        outputs_match = torch.equal(
            naive_output,
            cached_output
        )


        row = {

            "model":
                model_name,

            "parameters":
                parameter_count,

            "device":
                str(device),

            "prompt_length":
                prompt_length,

            "new_tokens":
                NEW_TOKENS,

            "no_cache_ms":
                naive_time * 1000,

            "model_ttft_ms":
                model_ttft * 1000,

            "decode_ms":
                decode_time * 1000,

            "tpot_ms":
                tpot_ms,

            "cached_total_ms":
                cached_total * 1000,

            "speedup_x":
                speedup,

            "no_cache_tokens_per_sec":
                NEW_TOKENS
                / naive_time,

            "cached_end_to_end_tokens_per_sec":
                NEW_TOKENS
                / cached_total,

            "decode_tokens_per_sec":
                decode_tokens_per_sec,

            "kv_cache_mib":
                cache_bytes
                / (1024 ** 2),

            "outputs_match":
                outputs_match,
        }


        rows.append(row)


        print(
            f"Prompt {prompt_length:4d} | "
            f"No-cache {naive_time * 1000:8.2f} ms | "
            f"Cache {cached_total * 1000:8.2f} ms | "
            f"Speedup {speedup:5.2f}x | "
            f"TTFT {model_ttft * 1000:6.2f} ms | "
            f"TPOT {tpot_ms:5.2f} ms | "
            f"KV {cache_bytes / (1024 ** 2):6.2f} MiB"
        )


    # ========================================================
    # CLEAN MODEL FROM MEMORY
    # ========================================================

    del model

    del tokenizer

    gc.collect()


    if device.type == "mps":

        torch.mps.empty_cache()


    elif device.type == "cuda":

        torch.cuda.empty_cache()


    return rows

all_results = []


for model_name in MODELS:

    rows = benchmark_model(
        model_name
    )

    all_results.extend(
        rows
    )


df = pd.DataFrame(
    all_results
)


output_path = Path(
    "benchmarks/"
    "real_model_scaling_results.csv"
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
print("=" * 70)
print("FINAL RESULTS")
print("=" * 70)


print(
    df.to_string(
        index=False
    )
)


print(
    "\nSaved:",
    output_path
)