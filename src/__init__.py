"""Double-Slit Simulation Laboratory package."""

import warnings

# Suppress harmless CuPy CUDA_PATH detection warning when using pip wheel packages
warnings.filterwarnings(
    "ignore",
    message="CUDA path could not be detected",
    category=UserWarning,
    module="cupy",
)
