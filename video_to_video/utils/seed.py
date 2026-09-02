import random
import numpy as np
import torch

TORCH_SPAN = 2**64  # torch accepts [-2**63, 2**64-1]
NUMPY_SPAN = 2**32  # np.random.seed accepts [0, 2**32-1]


def setup_seed(seed, deterministic=True, warn_only=True):
    """
    Seed every RNG safely.

    Any int (negative, huge, numpy scalar) is normalized per-library.
    Returns the canonical 64-bit seed actually used.
    """
    # --- normalize input -------------------------------------------------
    if isinstance(seed, (str, bytes)):
        seed = int.from_bytes(
            seed.encode() if isinstance(seed, str) else seed, "little"
        )
    seed = int(seed) % TORCH_SPAN  # -> [0, 2**64-1], always valid

    numpy_seed = seed % NUMPY_SPAN  # -> [0, 2**32-1]

    # --- seed each library ----------------------------------------------
    random.seed(seed)  # unbounded, full entropy
    np.random.seed(numpy_seed)  # 32-bit ceiling
    torch.manual_seed(seed)  # also seeds CPU + default CUDA

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # --- determinism knobs ----------------------------------------------
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        try:
            torch.use_deterministic_algorithms(True, warn_only=warn_only)
        except (AttributeError, TypeError):
            pass  # older torch: no warn_only kwarg, or API absent

    return seed
