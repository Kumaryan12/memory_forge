from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


INPUT = Path(
    "benchmarks/"
    "real_model_scaling_results.csv"
)

OUTPUT_DIR = Path(
    "benchmarks/plots"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


df = pd.read_csv(INPUT)


# ============================================================
# LATENCY
# ============================================================

for model_name in df["model"].unique():

    subset = df[
        df["model"] == model_name
    ]

    plt.figure(
        figsize=(8, 5)
    )

    plt.plot(
        subset["prompt_length"],
        subset["no_cache_ms"],
        marker="o",
        label="No KV cache"
    )

    plt.plot(
        subset["prompt_length"],
        subset["cached_total_ms"],
        marker="o",
        label="KV cache"
    )

    plt.xlabel(
        "Prompt Length (tokens)"
    )

    plt.ylabel(
        "Generation Time (ms)"
    )

    plt.title(
        f"{model_name}: "
        "KV Cache vs No Cache"
    )

    plt.legend()

    plt.grid(
        alpha=0.25
    )

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR
        / f"{model_name}_latency.png",
        dpi=200
    )

    plt.close()


print(
    "Latency plots generated."
)

plt.figure(
    figsize=(8, 5)
)


for model_name in df["model"].unique():

    subset = df[
        df["model"] == model_name
    ]

    plt.plot(
        subset["prompt_length"],
        subset["speedup_x"],
        marker="o",
        label=model_name
    )


plt.xlabel(
    "Prompt Length (tokens)"
)

plt.ylabel(
    "KV Cache Speedup (×)"
)

plt.title(
    "KV Cache Speedup "
    "vs Context Length"
)

plt.axhline(
    1.0,
    linestyle="--"
)

plt.legend()

plt.grid(
    alpha=0.25
)

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR
    / "kv_cache_speedup.png",
    dpi=200
)

plt.close()

plt.figure(
    figsize=(8, 5)
)


for model_name in df["model"].unique():

    subset = df[
        df["model"] == model_name
    ]

    plt.plot(
        subset["prompt_length"],
        subset["kv_cache_mib"],
        marker="o",
        label=model_name
    )


plt.xlabel(
    "Prompt Length (tokens)"
)

plt.ylabel(
    "KV Cache Memory (MiB)"
)

plt.title(
    "KV Cache Memory "
    "vs Context Length"
)

plt.legend()

plt.grid(
    alpha=0.25
)

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR
    / "kv_cache_memory.png",
    dpi=200
)

plt.close()

plt.figure(
    figsize=(8, 5)
)


for model_name in df["model"].unique():

    subset = df[
        df["model"] == model_name
    ]

    plt.plot(
        subset["prompt_length"],
        subset["model_ttft_ms"],
        marker="o",
        label=model_name
    )


plt.xlabel(
    "Prompt Length (tokens)"
)

plt.ylabel(
    "Model TTFT (ms)"
)

plt.title(
    "Prompt Length vs "
    "Model Time To First Token"
)

plt.legend()

plt.grid(
    alpha=0.25
)

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR
    / "ttft.png",
    dpi=200
)

plt.close()

plt.figure(
    figsize=(8, 5)
)


for model_name in df["model"].unique():

    subset = df[
        df["model"] == model_name
    ]

    plt.plot(
        subset["prompt_length"],
        subset["tpot_ms"],
        marker="o",
        label=model_name
    )


plt.xlabel(
    "Prompt Length (tokens)"
)

plt.ylabel(
    "TPOT (ms/token)"
)

plt.title(
    "Context Length vs "
    "Time Per Output Token"
)

plt.legend()

plt.grid(
    alpha=0.25
)

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR
    / "tpot.png",
    dpi=200
)

plt.close()


print(
    "All MemoryForge plots created."
)