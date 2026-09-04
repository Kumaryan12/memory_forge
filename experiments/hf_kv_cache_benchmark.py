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
# CONFIG
# ============================================================

MODEL_NAME = "distilgpt2"

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
print(
    f"\nLoading {MODEL_NAME}..."
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

def make_prompt(
    length
):

    # Use a real token repeatedly.
    #
    # Shape:
    # [1, length]

    token_id = (
        tokenizer.encode(
            " hello",
            add_special_tokens=False
        )[0]
    )


    prompt = torch.full(
        (1, length),
        token_id,
        dtype=torch.long,
        device=device
    )


    return prompt

@torch.inference_mode()
def generate_no_cache(
    prompt,
    new_tokens
):

    sequence = prompt.clone()


    for _ in range(
        new_tokens
    ):

        outputs = model(
            input_ids=sequence,
            use_cache=False
        )


        logits = outputs.logits[
            :,
            -1,
            :
        ]


        next_token = torch.argmax(
            logits,
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
    prompt,
    new_tokens
):

    sequence = prompt.clone()


    # --------------------------------------------------------
    # PREFILL
    # --------------------------------------------------------

    synchronize()

    prefill_start = (
        time.perf_counter()
    )


    outputs = model(
        input_ids=prompt,
        use_cache=True
    )


    synchronize()

    prefill_end = (
        time.perf_counter()
    )


    cache = (
        outputs.past_key_values
    )


    logits = outputs.logits[
        :,
        -1,
        :
    ]


    next_token = torch.argmax(
        logits,
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


        logits = outputs.logits[
            :,
            -1,
            :
        ]


        next_token = torch.argmax(
            logits,
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


    return (
        sequence,
        prefill_end
        - prefill_start,
        decode_end
        - decode_start,
        cache
    )

results = []


print("\nStarting real-model benchmark...\n")


for prompt_length in PROMPT_LENGTHS:

    prompt = make_prompt(
        prompt_length
    )


    # ========================================================
    # WARM-UP
    # ========================================================

    _ = generate_no_cache(
        prompt,
        2
    )


    _ = generate_with_cache(
        prompt,
        2
    )


    naive_times = []

    cached_total_times = []

    prefill_times = []

    decode_times = []


    # ========================================================
    # RUNS
    # ========================================================

    for _ in range(REPEATS):

        # ----------------------------------------------------
        # NO CACHE
        # ----------------------------------------------------

        synchronize()

        start = time.perf_counter()


        naive_output = (
            generate_no_cache(
                prompt,
                NEW_TOKENS
            )
        )


        synchronize()

        end = time.perf_counter()


        naive_times.append(
            end - start
        )


        # ----------------------------------------------------
        # WITH CACHE
        # ----------------------------------------------------

        (
            cached_output,
            prefill_time,
            decode_time,
            cache
        ) = generate_with_cache(
            prompt,
            NEW_TOKENS
        )


        prefill_times.append(
            prefill_time
        )

        decode_times.append(
            decode_time
        )

        cached_total_times.append(
            prefill_time
            + decode_time
        )


    # ========================================================
    # CORRECTNESS
    # ========================================================

    same_output = torch.equal(
        naive_output,
        cached_output
    )


    # ========================================================
    # MEDIANS
    # ========================================================

    naive_time = statistics.median(
        naive_times
    )


    cached_time = statistics.median(
        cached_total_times
    )


    prefill_time = statistics.median(
        prefill_times
    )


    decode_time = statistics.median(
        decode_times
    )


    speedup = (
        naive_time
        / cached_time
    )


    row = {

        "model":
            MODEL_NAME,

        "device":
            str(device),

        "prompt_length":
            prompt_length,

        "new_tokens":
            NEW_TOKENS,

        "naive_ms":
            naive_time * 1000,

        "prefill_ms":
            prefill_time * 1000,

        "decode_ms":
            decode_time * 1000,

        "cached_total_ms":
            cached_time * 1000,

        "speedup_x":
            speedup,

        "naive_tokens_per_sec":
            NEW_TOKENS
            / naive_time,

        "cached_tokens_per_sec":
            NEW_TOKENS
            / cached_time,

        "outputs_match":
            same_output,
    }


    results.append(
        row
    )


    print(
        f"Prompt {prompt_length:4d} | "
        f"No cache "
        f"{naive_time * 1000:8.2f} ms | "
        f"Cache "
        f"{cached_time * 1000:8.2f} ms | "
        f"Speedup "
        f"{speedup:5.2f}x | "
        f"Match: "
        f"{same_output}"
    )

df = pd.DataFrame(
    results
)


output_path = Path(
    "benchmarks/"
    "hf_kv_cache_results.csv"
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