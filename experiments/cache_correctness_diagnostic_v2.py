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


print("=" * 72)
print("MEMORYFORGE — CACHE CORRECTNESS DIAGNOSTIC V2")
print("=" * 72)

print("Device:", device)
print("Model :", MODEL_NAME)


def synchronize():

    if device.type == "cuda":
        torch.cuda.synchronize()

    elif device.type == "mps":
        torch.mps.synchronize()


# ============================================================
# LOAD MODEL
# ============================================================

print("\nLoading tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME
)


print("Loading model...")

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
# CREATE CONTROLLED PROMPT
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
# CACHE LENGTH
# ============================================================

def get_cache_length(cache):
    """
    Return the number of tokens currently stored
    in the KV cache.

    Supports:
    - New Hugging Face DynamicCache API
    - Older DynamicCache API
    - Legacy tuple cache
    """

    if cache is None:
        return 0


    # ========================================================
    # Preferred public API
    # ========================================================

    if hasattr(cache, "get_seq_length"):

        try:
            return int(
                cache.get_seq_length()
            )
        except TypeError:
            return int(
                cache.get_seq_length(0)
            )


    # ========================================================
    # New DynamicCache:
    #
    # cache.layers[0].keys
    # ========================================================

    if hasattr(cache, "layers"):

        if len(cache.layers) == 0:
            return 0

        layer = cache.layers[0]

        if hasattr(
            layer,
            "get_seq_length"
        ):
            return int(
                layer.get_seq_length()
            )

        if (
            hasattr(layer, "keys")
            and layer.keys is not None
        ):
            return int(
                layer.keys.shape[-2]
            )


    # ========================================================
    # Older DynamicCache
    # ========================================================

    if hasattr(cache, "key_cache"):

        return int(
            cache.key_cache[0]
            .shape[-2]
        )


    # ========================================================
    # Legacy tuple
    # ========================================================

    try:

        return int(
            cache[0][0]
            .shape[-2]
        )

    except Exception as exc:

        raise TypeError(
            "Unsupported KV cache format. "
            f"Cache type: {type(cache)}"
        ) from exc

# ============================================================
# CACHE INFO
# ============================================================

def print_cache_info(cache):
    """
    Display dtype and tensor shape of
    the first layer's K/V cache.

    Compatible with multiple Hugging Face
    cache API versions.
    """

    if cache is None:

        print("No cache returned.")
        return


    K = None
    V = None


    # ========================================================
    # NEW HUGGING FACE DYNAMIC CACHE
    #
    # cache.layers[0].keys
    # cache.layers[0].values
    # ========================================================

    if hasattr(cache, "layers"):

        if len(cache.layers) == 0:

            print(
                "Cache exists but contains "
                "no layers."
            )

            return


        first_layer = (
            cache.layers[0]
        )


        if (
            hasattr(
                first_layer,
                "keys"
            )
            and hasattr(
                first_layer,
                "values"
            )
        ):

            K = first_layer.keys

            V = first_layer.values


    # ========================================================
    # OLDER DYNAMIC CACHE
    # ========================================================

    if (
        K is None
        and hasattr(
            cache,
            "key_cache"
        )
    ):

        K = cache.key_cache[0]

        V = cache.value_cache[0]


    # ========================================================
    # LEGACY TUPLE CACHE
    # ========================================================

    if K is None:

        try:

            K = cache[0][0]

            V = cache[0][1]

        except Exception as exc:

            print(
                "\nCould not inspect cache."
            )

            print(
                "Cache type:",
                type(cache)
            )

            print(
                "Available attributes:"
            )

            print(
                [
                    name
                    for name in dir(cache)
                    if not name.startswith("_")
                ]
            )

            raise RuntimeError(
                "Unsupported cache representation."
            ) from exc


    # ========================================================
    # DISPLAY
    # ========================================================

    print(
        "Cache type     :",
        type(cache).__name__
    )

    print(
        "KV dtype       :",
        K.dtype
    )

    print(
        "K shape        :",
        tuple(K.shape)
    )

    print(
        "V shape        :",
        tuple(V.shape)
    )

    print(
        "Cached tokens  :",
        get_cache_length(cache)
    )
# ============================================================
# TOP K
# ============================================================

def display_top_k(
    logits,
    title,
    k=5
):

    values, indices = torch.topk(
        logits,
        k=k,
        dim=-1
    )


    print(f"\n{title}")


    for rank in range(k):

        token_id = (
            indices[0, rank]
            .item()
        )

        logit = (
            values[0, rank]
            .item()
        )

        text = tokenizer.decode(
            [token_id]
        )


        print(
            f"#{rank + 1} "
            f"id={token_id:6d} | "
            f"logit={logit:10.6f} | "
            f"{text!r}"
        )


# ============================================================
# DIAGNOSTIC
# ============================================================

@torch.inference_mode()
def diagnose(
    prompt,
    max_new_tokens
):

    prompt_length = (
        prompt.shape[1]
    )


    print(
        "\nPrompt length:",
        prompt_length
    )


    # ========================================================
    # PREFILL
    # ========================================================

    prompt_attention_mask = torch.ones(
        (
            1,
            prompt_length
        ),
        dtype=torch.long,
        device=device
    )


    prompt_cache_position = torch.arange(
        0,
        prompt_length,
        dtype=torch.long,
        device=device
    )


    synchronize()


    cached_outputs = model(

        input_ids=prompt,

        attention_mask=(
            prompt_attention_mask
        ),

        cache_position=(
            prompt_cache_position
        ),

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


    print_cache_info(cache)


    # Full-context branch
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


    # ========================================================
    # TOKEN-BY-TOKEN TEST
    # ========================================================

    for step in range(
        max_new_tokens
    ):

        current_length = (
            full_sequence.shape[1]
        )


        # ====================================================
        # FULL CONTEXT
        # ====================================================

        full_attention_mask = torch.ones(

            (
                1,
                current_length
            ),

            dtype=torch.long,

            device=device
        )


        synchronize()


        naive_outputs = model(

            input_ids=full_sequence,

            attention_mask=(
                full_attention_mask
            ),

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
        # NUMERICAL ERROR
        # ====================================================

        diff = (
            naive_logits
            - cached_logits
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

            naive_logits,

            cached_logits,

            rtol=RTOL,

            atol=ATOL
        )


        allclose_results.append(
            close
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
        # TOP-1 / TOP-2 MARGIN
        # ====================================================

        top_values, _ = torch.topk(

            naive_logits,

            k=2,

            dim=-1
        )


        top1_margin = (

            top_values[0, 0]

            -

            top_values[0, 1]

        ).item()


        print(
            f"\nStep {step:02d}"
        )

        print("-" * 56)

        print(
            "Context length       :",
            current_length
        )

        print(
            "Cache length         :",
            get_cache_length(cache)
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
            "torch.allclose       :",
            close
        )

        print(
            f"Top-1 margin         : "
            f"{top1_margin:.8f}"
        )

        print(
            "Naive token          :",
            naive_token.item(),
            repr(
                tokenizer.decode(
                    [naive_token.item()]
                )
            )
        )

        print(
            "Cached token         :",
            cached_token.item(),
            repr(
                tokenizer.decode(
                    [cached_token.item()]
                )
            )
        )

        print(
            "Same token           :",
            same_token
        )


        # ====================================================
        # STOP IF THEY CHOOSE DIFFERENT TOKENS
        # ====================================================

        if not same_token:

            first_mismatch = step


            print(
                "\n"
                "!!! GREEDY TOKEN DIVERGENCE !!!"
            )


            display_top_k(
                naive_logits,
                "Full-context top 5"
            )


            display_top_k(
                cached_logits,
                "Cached top 5"
            )


            break


        # ====================================================
        # APPEND AGREED TOKEN TO FULL CONTEXT
        # ====================================================

        full_sequence = torch.cat(

            [
                full_sequence,
                naive_token
            ],

            dim=1
        )


        # ====================================================
        # PREPARE CACHED DECODE POSITION
        # ====================================================

        cache_length = (
            get_cache_length(
                cache
            )
        )


        # New token belongs exactly after
        # all previously cached tokens.
        cache_position = torch.tensor(

            [cache_length],

            dtype=torch.long,

            device=device
        )


        # Attention mask must cover:
        #
        # cached tokens + current token

        cached_attention_mask = torch.ones(

            (
                1,
                cache_length + 1
            ),

            dtype=torch.long,

            device=device
        )


        # ====================================================
        # CACHED DECODE
        # ====================================================

        synchronize()


        cached_outputs = model(

            input_ids=cached_token,

            attention_mask=(
                cached_attention_mask
            ),

            past_key_values=cache,

            cache_position=(
                cache_position
            ),

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

    steps_tested = len(
        max_diffs
    )


    match_rate = (

        sum(top1_matches)

        / len(top1_matches)

    )


    allclose_rate = (

        sum(allclose_results)

        / len(allclose_results)

    )


    print(
        "\n"
        + "=" * 72
    )

    print(
        "SUMMARY"
    )

    print(
        "=" * 72
    )


    print(
        "Steps tested:",
        steps_tested
    )


    print(
        f"Maximum logit difference: "
        f"{max(max_diffs):.8f}"
    )


    print(
        f"Average mean difference: "
        f"{sum(mean_diffs) / len(mean_diffs):.8f}"
    )


    print(
        f"Top-1 match rate: "
        f"{match_rate * 100:.2f}%"
    )


    print(
        f"Allclose rate: "
        f"{allclose_rate * 100:.2f}%"
    )


    print(
        "First mismatch:",
        first_mismatch
    )


    return {

        "prompt_length":
            prompt_length,

        "steps_tested":
            steps_tested,

        "max_diff":
            max(max_diffs),

        "mean_diff":
            (
                sum(mean_diffs)
                / len(mean_diffs)
            ),

        "top1_match_rate":
            match_rate,

        "allclose_rate":
            allclose_rate,

        "first_mismatch":
            first_mismatch,
    }


# ============================================================
# RUN
# ============================================================

results = []


for length in PROMPT_LENGTHS:

    print(
        "\n\n"
        + "#" * 72
    )

    print(
        f"PROMPT LENGTH {length}"
    )

    print(
        "#" * 72
    )


    prompt = make_prompt(
        length
    )


    result = diagnose(

        prompt,

        MAX_NEW_TOKENS
    )


    results.append(
        result
    )


# ============================================================
# FINAL SUMMARY
# ============================================================

print(
    "\n\n"
    + "=" * 72
)

print(
    "FINAL COMPARISON"
)

print(
    "=" * 72
)


for result in results:

    mismatch = (
        result[
            "first_mismatch"
        ]
    )


    print(

        f"Prompt "
        f"{result['prompt_length']:4d} | "

        f"Max diff "
        f"{result['max_diff']:.6f} | "

        f"Top1 "
        f"{result['top1_match_rate'] * 100:6.2f}% | "

        f"Allclose "
        f"{result['allclose_rate'] * 100:6.2f}% | "

        f"Mismatch "
        f"{mismatch}"

    )