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

MAX_NEW_TOKENS = 16

RTOL = 1e-3
ATOL = 1e-3


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


print("=" * 70)
print("MEMORYFORGE — KV CACHE CORRECTNESS DIAGNOSTIC")
print("=" * 70)

print("Device:", device)
print("Model:", MODEL_NAME)


# ============================================================
# DEVICE SYNCHRONIZATION
# ============================================================

def synchronize():

    if device.type == "cuda":

        torch.cuda.synchronize()

    elif device.type == "mps":

        torch.mps.synchronize()


# ============================================================
# LOAD TOKENIZER
# ============================================================

print("\nLoading tokenizer...")


tokenizer = (
    AutoTokenizer
    .from_pretrained(
        MODEL_NAME
    )
)


# ============================================================
# LOAD MODEL
# ============================================================

print("Loading model...")


model = (
    AutoModelForCausalLM
    .from_pretrained(
        MODEL_NAME
    )
    .to(device)
)


model.eval()


# ============================================================
# MODEL INFORMATION
# ============================================================

parameter_count = sum(
    p.numel()
    for p in model.parameters()
)


model_dtype = next(
    model.parameters()
).dtype


print(
    "Parameters:",
    f"{parameter_count:,}"
)

print(
    "Model dtype:",
    model_dtype
)


# ============================================================
# CONTROLLED PROMPT
# ============================================================

def make_prompt(
    length
):
    """
    Create a deterministic prompt containing
    exactly `length` tokens.

    Repeating one token keeps the experiment
    controlled: the changing variable is
    sequence length, not prompt content.
    """

    token_ids = tokenizer.encode(
        " hello",
        add_special_tokens=False
    )


    token_id = token_ids[0]


    prompt = torch.full(
        (
            1,
            length
        ),
        token_id,
        dtype=torch.long,
        device=device
    )


    return prompt


# ============================================================
# CACHE DTYPE INFORMATION
# ============================================================

def print_cache_information(
    cache
):

    if cache is None:

        print(
            "No cache returned."
        )

        return


    try:

        # -----------------------------------------------
        # Modern Hugging Face DynamicCache
        # -----------------------------------------------

        if (
            hasattr(
                cache,
                "key_cache"
            )
            and hasattr(
                cache,
                "value_cache"
            )
        ):

            K = cache.key_cache[0]

            V = cache.value_cache[0]


        # -----------------------------------------------
        # Legacy tuple cache
        # -----------------------------------------------

        else:

            K = cache[0][0]

            V = cache[0][1]


        print(
            "First K dtype:",
            K.dtype
        )

        print(
            "First V dtype:",
            V.dtype
        )

        print(
            "First K shape:",
            tuple(K.shape)
        )

        print(
            "First V shape:",
            tuple(V.shape)
        )


    except Exception as exc:

        print(
            "Could not inspect cache:",
            exc
        )


# ============================================================
# TOP-K DISPLAY
# ============================================================

def show_top_tokens(
    logits,
    title,
    k=5
):

    values, indices = torch.topk(
        logits,
        k=k,
        dim=-1
    )


    print(
        f"\n{title}"
    )


    for rank in range(k):

        token_id = (
            indices[
                0,
                rank
            ]
            .item()
        )


        value = (
            values[
                0,
                rank
            ]
            .item()
        )


        decoded = tokenizer.decode(
            [token_id]
        )


        print(
            f"  #{rank + 1} | "
            f"id={token_id:6d} | "
            f"logit={value:10.6f} | "
            f"token={decoded!r}"
        )


# ============================================================
# CACHE CORRECTNESS DIAGNOSTIC
# ============================================================

@torch.inference_mode()
def diagnose_cache(
    prompt,
    max_new_tokens
):

    prompt_length = (
        prompt.shape[1]
    )


    print(
        "\nInitial prompt length:",
        prompt_length
    )


    # ========================================================
    # CACHED PREFILL
    # ========================================================

    synchronize()


    cached_outputs = model(
        input_ids=prompt,
        use_cache=True
    )


    synchronize()


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
    # SHOW CACHE INFORMATION
    # ========================================================

    print_cache_information(
        cache
    )


    # ========================================================
    # FULL SEQUENCE FOR NAIVE PATH
    # ========================================================

    full_sequence = (
        prompt.clone()
    )


    # ========================================================
    # STATISTICS
    # ========================================================

    max_diffs = []

    mean_diffs = []

    top1_matches = []

    allclose_results = []


    first_mismatch_step = None


    # ========================================================
    # DECODE STEP BY STEP
    # ========================================================

    for step in range(
        max_new_tokens
    ):

        # ====================================================
        # NAIVE PATH
        #
        # Recalculate the ENTIRE context.
        # ====================================================

        synchronize()


        naive_outputs = model(
            input_ids=full_sequence,
            use_cache=False
        )


        synchronize()


        naive_logits = (
            naive_outputs
            .logits[
                :,
                -1,
                :
            ]
        )


        # ====================================================
        # NUMERICAL DIFFERENCE
        # ====================================================

        difference = (
            naive_logits
            - cached_logits
        ).abs()


        max_diff = (
            difference
            .max()
            .item()
        )


        mean_diff = (
            difference
            .mean()
            .item()
        )


        max_diffs.append(
            max_diff
        )


        mean_diffs.append(
            mean_diff
        )


        # ====================================================
        # ALLCLOSE TEST
        # ====================================================

        logits_close = torch.allclose(

            naive_logits,

            cached_logits,

            rtol=RTOL,

            atol=ATOL

        )


        allclose_results.append(
            logits_close
        )


        # ====================================================
        # GREEDY TOKEN
        # ====================================================

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


        # ====================================================
        # TOP-1 MARGIN
        # ====================================================

        top_values, top_indices = torch.topk(
            naive_logits,
            k=2,
            dim=-1
        )


        top1_margin = (

            top_values[
                0,
                0
            ]

            -

            top_values[
                0,
                1
            ]

        ).item()


        # ====================================================
        # DECODE TOKENS FOR DISPLAY
        # ====================================================

        naive_token_id = (
            naive_token.item()
        )


        cached_token_id = (
            cached_token.item()
        )


        naive_text = tokenizer.decode(
            [naive_token_id]
        )


        cached_text = tokenizer.decode(
            [cached_token_id]
        )


        # ====================================================
        # PRINT STEP RESULT
        # ====================================================

        print(
            f"\nStep {step:02d}"
        )

        print(
            "-" * 50
        )

        print(
            f"Context length       : "
            f"{full_sequence.shape[1]}"
        )

        print(
            f"Max logit difference : "
            f"{max_diff:.8f}"
        )

        print(
            f"Mean logit difference: "
            f"{mean_diff:.8f}"
        )

        print(
            f"torch.allclose       : "
            f"{logits_close}"
        )

        print(
            f"Top-1 margin         : "
            f"{top1_margin:.8f}"
        )

        print(
            f"Naive token          : "
            f"{naive_token_id} "
            f"{naive_text!r}"
        )

        print(
            f"Cached token         : "
            f"{cached_token_id} "
            f"{cached_text!r}"
        )

        print(
            f"Same token           : "
            f"{same_token}"
        )


        # ====================================================
        # IF TOKEN DIVERGED
        # ====================================================

        if not same_token:

            if (
                first_mismatch_step
                is None
            ):

                first_mismatch_step = (
                    step
                )


            print(
                "\n!!! TOKEN DIVERGENCE DETECTED !!!"
            )


            show_top_tokens(
                naive_logits,
                "Naive top predictions"
            )


            show_top_tokens(
                cached_logits,
                "Cached top predictions"
            )


            # ------------------------------------------------
            # Important:
            #
            # Stop here.
            #
            # Once sequences diverge, future logits would
            # represent DIFFERENT contexts, so comparing
            # them would no longer isolate cache effects.
            # ------------------------------------------------

            break


        # ====================================================
        # APPEND AGREED TOKEN
        # ====================================================

        full_sequence = torch.cat(
            [
                full_sequence,
                naive_token
            ],
            dim=1
        )


        # ====================================================
        # CACHED NEXT STEP
        #
        # Only process the new token.
        # ====================================================

        synchronize()


        cached_outputs = model(
            input_ids=cached_token,
            past_key_values=cache,
            use_cache=True
        )


        synchronize()


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

    print(
        "\n"
        + "=" * 70
    )

    print(
        "DIAGNOSTIC SUMMARY"
    )

    print(
        "=" * 70
    )


    completed_steps = len(
        max_diffs
    )


    print(
        "Steps tested:",
        completed_steps
    )


    if max_diffs:

        print(
            "Maximum logit difference:",
            f"{max(max_diffs):.8f}"
        )


        print(
            "Average mean-logit difference:",
            f"{sum(mean_diffs) / len(mean_diffs):.8f}"
        )


    top1_match_rate = (

        sum(
            1
            for result
            in top1_matches
            if result
        )

        / len(top1_matches)

        if top1_matches
        else 0

    )


    allclose_rate = (

        sum(
            1
            for result
            in allclose_results
            if result
        )

        / len(allclose_results)

        if allclose_results
        else 0

    )


    print(
        "Top-1 token match rate:",
        f"{top1_match_rate * 100:.2f}%"
    )


    print(
        "Allclose rate:",
        f"{allclose_rate * 100:.2f}%"
    )


    if (
        first_mismatch_step
        is None
    ):

        print(
            "\n✓ No greedy-token divergence detected."
        )

    else:

        print(
            "\n✗ First greedy-token divergence "
            f"at decode step "
            f"{first_mismatch_step}."
        )


    return {

        "prompt_length":
            prompt_length,

        "steps_tested":
            completed_steps,

        "max_logit_diff":
            (
                max(max_diffs)
                if max_diffs
                else 0
            ),

        "mean_logit_diff":
            (
                sum(mean_diffs)
                / len(mean_diffs)
                if mean_diffs
                else 0
            ),

        "top1_match_rate":
            top1_match_rate,

        "allclose_rate":
            allclose_rate,

        "first_mismatch_step":
            first_mismatch_step,
    }


# ============================================================
# RUN DIAGNOSTICS
# ============================================================

all_results = []


for prompt_length in PROMPT_LENGTHS:

    print(
        "\n\n"
        + "#" * 70
    )

    print(
        f"PROMPT LENGTH: "
        f"{prompt_length}"
    )

    print(
        "#" * 70
    )


    prompt = make_prompt(
        prompt_length
    )


    result = diagnose_cache(

        prompt=prompt,

        max_new_tokens=(
            MAX_NEW_TOKENS
        )

    )


    all_results.append(
        result
    )


# ============================================================
# FINAL CROSS-CONTEXT SUMMARY
# ============================================================

print(
    "\n\n"
    + "=" * 70
)

print(
    "MEMORYFORGE — FINAL CACHE CORRECTNESS SUMMARY"
)

print(
    "=" * 70
)


for result in all_results:

    mismatch = (
        result[
            "first_mismatch_step"
        ]
    )


    mismatch_text = (

        "None"

        if mismatch is None

        else str(mismatch)

    )


    print(

        f"Prompt "
        f"{result['prompt_length']:4d} | "

        f"Max diff "
        f"{result['max_logit_diff']:.6f} | "

        f"Top1 match "
        f"{result['top1_match_rate'] * 100:6.2f}% | "

        f"Allclose "
        f"{result['allclose_rate'] * 100:6.2f}% | "

        f"First mismatch "
        f"{mismatch_text}"

    )