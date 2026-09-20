from .reward import score_trajectory

# Torch is only required when the GRPO advantage adapter is used.  Keeping this
# import lazy lets data preparation, retrieval smoke tests, and reward audits
# run on a lightweight CPU environment before the training stack is installed.
try:
    from .advantage import grouped_advantages
except ModuleNotFoundError as exc:
    if exc.name != "torch":
        raise
    grouped_advantages = None
