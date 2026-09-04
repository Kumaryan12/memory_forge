import gc

import pandas as pd
import torch

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)


# ============================================================
# CONFIG
# ============================================================

MODEL_NAME = "Qwen/Qwen2.5-0.5B"

PROMPT_LENGTHS = [
    128,
    256,
]

MAX_NEW_TOKENS = 12

RTOL = 1e-3
ATOL = 1e-3


EXPERIMENTS = [
    {
        "name": "BF16_SDPA",
        "dtype": torch.bfloat16,
        "attention": "sdpa",
    },
    {
        "name": "BF16_EAGER",
        "dtype": torch.bfloat16,
        "attention": "eager",
    },
    {
        "name": "FP32_SDPA",
        "dtype": torch.float32,
        "attention": "sdpa",
    },
    {
        "name": "FP32_EAGER",
        "dtype": torch.float32,
        "attention": "eager",
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
print("MEMORYFORGE — PRECISION × ATTENTION CORRECTNESS MATRIX")
print("=" * 75)

print("Device:", device)
print("Model :", MODEL_NAME)


torch.manual_seed(42)


# ============================================================
# TOKENIZER
# ============================================================

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME
)


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
# SYNCHRONIZATION
# ============================================================

def synchronize():

    if device.type == "cuda":
        torch.cuda.synchronize()

    elif device.type == "mps":
        torch.mps.synchronize()


# ============================================================
# CACHE LENGTH
# ============================================================

def get_cache_length(cache):

    if cache is None:
        return 0

    if hasattr(
        cache,
        "get_seq_length"
    ):
        return int(
            cache.get_seq_length()
        )

    if hasattr(
        cache,
        "layers"
    ):

        first_layer = (
            cache.layers[0]
        )

        return int(
            first_layer.keys.shape[-2]
        )

    if hasattr(
        cache,
        "key_cache"
    ):

        return int(
            cache.key_cache[0]
            .shape[-2]
        )

    return int(
        cache[0][0]
        .shape[-2]
    )


# ============================================================
# LOAD MODEL
# ============================================================

def load_model(
    dtype,
    attention
):

    print(
        "\nLoading:",
        dtype,
        "| attention:",
        attention
    )

    model = (
        AutoModelForCausalLM
        .from_pretrained(
            MODEL_NAME,
            dtype=dtype,
            attn_implementation=attention,
        )
        .to(device)
    )

    model.eval()

    return model


# ============================================================
# DIAGNOSTIC
# ============================================================

@torch.inference_mode()
def diagnose(
    model,
    prompt,
    max_new_tokens
):

    prompt_length = (
        prompt.shape[1]
    )


    # ========================================================
    # CACHED PREFILL
    # ========================================================

    attention_mask = torch.ones(
        (
            1,
            prompt_length
        ),
        dtype=torch.long,
        device=device
    )


    cache_position = torch.arange(
        prompt_length,
        dtype=torch.long,
        device=device
    )


    cached_outputs = model(
        input_ids=prompt,
        attention_mask=attention_mask,
        cache_position=cache_position,
        use_cache=True,
    )


    cache = (
        cached_outputs
        .past_key_values
    )


    cached_logits = (
        cached_outputs
        .logits[
            :,
            -1,
            :
        ]
    )


    full_sequence = (
        prompt.clone()
    )


    # ========================================================
    # METRICS
    # ========================================================

    max_diffs = []

    mean_diffs = []

    top1_matches = []

    allclose_results = []

    first_mismatch = None

    mismatch_margin = None


    # ========================================================
    # DECODE
    # ========================================================

    for step in range(
        max_new_tokens
    ):

        current_length = (
            full_sequence.shape[1]
        )


        # ----------------------------------------------------
        # FULL CONTEXT
        # ----------------------------------------------------

        full_mask = torch.ones(
            (
                1,
                current_length
            ),
            dtype=torch.long,
            device=device
        )


        naive_outputs = model(
            input_ids=full_sequence,
            attention_mask=full_mask,
            use_cache=False,
        )


        naive_logits = (
            naive_outputs
            .logits[
                :,
                -1,
                :
            ]
        )


        # ----------------------------------------------------
        # DIFFERENCE
        # ----------------------------------------------------

        diff = (
            naive_logits.float()
            - cached_logits.float()
        ).abs()


        max_diff = (
            diff.max().item()
        )


        mean_diff = (
            diff.mean().item()
        )


        max_diffs.append(
            max_diff
        )

        mean_diffs.append(
            mean_diff
        )


        close = torch.allclose(
            naive_logits.float(),
            cached_logits.float(),
            rtol=RTOL,
            atol=ATOL,
        )


        allclose_results.append(
            close
        )


        # ----------------------------------------------------
        # TOP-1
        # ----------------------------------------------------

        naive_token = torch.argmax(
            naive_logits,
            dim=-1,
            keepdim=True
        )


        cached_token = torch.argmax(
            cached_logits,
            dim=-1,
            keepdim=True
        )


        same_token = torch.equal(
            naive_token,
            cached_token
        )


        top1_matches.append(
            same_token
        )


        # ----------------------------------------------------
        # TOP-1 MARGIN
        # ----------------------------------------------------

        top_values, _ = torch.topk(
            naive_logits.float(),
            k=2,
            dim=-1
        )


        margin = (
            top_values[0, 0]
            - top_values[0, 1]
        ).item()


        # ----------------------------------------------------
        # STOP AT FIRST DIVERGENCE
        # ----------------------------------------------------

        if not same_token:

            first_mismatch = (
                step
            )

            mismatch_margin = (
                margin
            )

            print(
                f"\nMismatch at step {step}"
            )

            print(
                "Context:",
                current_length
            )

            print(
                "Max logit diff:",
                max_diff
            )

            print(
                "Mean logit diff:",
                mean_diff
            )

            print(
                "Top-1 margin:",
                margin
            )

            print(
                "Naive token:",
                naive_token.item()
            )

            print(
                "Cached token:",
                cached_token.item()
            )

            break


        # ----------------------------------------------------
        # APPEND AGREED TOKEN
        # ----------------------------------------------------

        full_sequence = torch.cat(
            [
                full_sequence,
                naive_token
            ],
            dim=1
        )


        # ----------------------------------------------------
        # CACHED NEXT TOKEN
        # ----------------------------------------------------

        cache_length = (
            get_cache_length(
                cache
            )
        )


        next_cache_position = torch.tensor(
            [cache_length],
            dtype=torch.long,
            device=device
        )


        cached_mask = torch.ones(
            (
                1,
                cache_length + 1
            ),
            dtype=torch.long,
            device=device
        )


        cached_outputs = model(
            input_ids=cached_token,
            attention_mask=cached_mask,
            past_key_values=cache,
            cache_position=(
                next_cache_position
            ),
            use_cache=True,
        )


        cache = (
            cached_outputs
            .past_key_values
        )


        cached_logits = (
            cached_outputs
            .logits[
                :,
                -1,
                :
            ]
        )


    # ========================================================
    # SUMMARY
    # ========================================================

    steps = len(
        max_diffs
    )


    top1_rate = (
        sum(top1_matches)
        / len(top1_matches)
    )


    allclose_rate = (
        sum(allclose_results)
        / len(allclose_results)
    )


    return {
        "prompt_length":
            prompt_length,

        "steps_tested":
            steps,

        "max_logit_diff":
            max(max_diffs),

        "mean_logit_diff":
            (
                sum(mean_diffs)
                / len(mean_diffs)
            ),

        "top1_match_rate":
            top1_rate,

        "allclose_rate":
            allclose_rate,

        "first_mismatch":
            first_mismatch,

        "mismatch_top1_margin":
            mismatch_margin,
    }


# ============================================================
# RUN MATRIX
# ============================================================

results = []


for experiment in EXPERIMENTS:

    name = (
        experiment["name"]
    )

    dtype = (
        experiment["dtype"]
    )

    attention = (
        experiment["attention"]
    )


    print(
        "\n\n"
        + "#" * 75
    )

    print(
        "EXPERIMENT:",
        name
    )

    print(
        "#" * 75
    )


    model = load_model(
        dtype=dtype,
        attention=attention
    )


    actual_dtype = next(
        model.parameters()
    ).dtype


    print(
        "Actual model dtype:",
        actual_dtype
    )


    for prompt_length in (
        PROMPT_LENGTHS
    ):

        print(
            "\nPrompt:",
            prompt_length
        )


        prompt = make_prompt(
            prompt_length
        )


        result = diagnose(
            model=model,
            prompt=prompt,
            max_new_tokens=(
                MAX_NEW_TOKENS
            ),
        )


        result[
            "experiment"
        ] = name

        result[
            "dtype"
        ] = str(
            actual_dtype
        )

        result[
            "attention_backend"
        ] = attention


        results.append(
            result
        )


        print(
            f"Top1 match: "
            f"{result['top1_match_rate'] * 100:.2f}%"
        )

        print(
            f"Max diff: "
            f"{result['max_logit_diff']:.6f}"
        )

        print(
            "First mismatch:",
            result["first_mismatch"]
        )


    # ========================================================
    # FREE MEMORY
    # ========================================================

    del model

    gc.collect()


    if device.type == "mps":

        torch.mps.empty_cache()


    elif device.type == "cuda":

        torch.cuda.empty_cache()


# ============================================================
# SAVE
# ============================================================

df = pd.DataFrame(
    results
)


output_path = (
    "benchmarks/"
    "precision_attention_correctness.csv"
)


df.to_csv(
    output_path,
    index=False
)


print(
    "\n\n"
    + "=" * 75
)

print(
    "FINAL CORRECTNESS MATRIX"
)

print(
    "=" * 75
)


display_columns = [
    "experiment",
    "prompt_length",
    "max_logit_diff",
    "mean_logit_diff",
    "top1_match_rate",
    "allclose_rate",
    "first_mismatch",
    "mismatch_top1_margin",
]


print(
    df[
        display_columns
    ].to_string(
        index=False
    )
)


print(
    "\nSaved:",
    output_path
)