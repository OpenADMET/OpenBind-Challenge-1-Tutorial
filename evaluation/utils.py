"""Utility functions for the OpenBind blind challenge 1."""

from functools import lru_cache

import numpy as np

BOOTSTRAP_SEED = 0

@lru_cache(maxsize=3)
def bootstrap_sampling(
    original_dataset_size: int, n_bootstrap_repeats: int = 1000
) -> np.ndarray:
    """Generate bootstrap sample indices for a dataset of a given size.

    The random seed is fixed so all submissions are evaluated on the same bootstrap
    samples. Best practices for bootstrap sampling involve sampling the same number of
    samples as the original dataset, with replacement, at least 1000 times.

    Args:
        original_dataset_size (int): The size of the original dataset.
        n_bootstrap_repeats (int): The number of bootstrap samples to generate.
                                   Default is 1000.

    Returns:
        np.ndarray: An array of bootstrap sample indices.

    """
    rng = np.random.default_rng(seed=BOOTSTRAP_SEED)
    return rng.choice(
        original_dataset_size,
        size=(n_bootstrap_repeats, original_dataset_size),
        replace=True,
    )
