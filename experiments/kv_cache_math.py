def kv_cache_memory(
    num_layers,
    sequence_length,
    hidden_size,
    bytes_per_value
):
    """
    Approximate KV-cache memory.

    Each layer stores:
    - Key tensor
    - Value tensor

    So we multiply by 2.
    """

    total_bytes = (
        2
        * num_layers
        * sequence_length
        * hidden_size
        * bytes_per_value
    )

    memory_mb = total_bytes / (1024 ** 2)

    return memory_mb


num_layers = 12
hidden_size = 768

print("GPT-style KV Cache Estimate\n")

for sequence_length in [
    512,
    1024,
    2048,
    4096,
    8192
]:

    memory = kv_cache_memory(
        num_layers=num_layers,
        sequence_length=sequence_length,
        hidden_size=hidden_size,
        bytes_per_value=2
    )

    print(
        f"Sequence length: {sequence_length:5d} | "
        f"KV cache: {memory:.2f} MB"
    )