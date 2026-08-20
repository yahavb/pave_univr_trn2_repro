"""
UniVR Utilities Module

Path setup, model validation, and tensor operations for UniVR.
"""

import logging
import os
import sys
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


def setup_univr_paths(base_dir: Optional[str] = None, model_type: Optional[str] = None) -> None:
    """Add UniVR and required subdirectories to the Python path.

    Args:
        base_dir: Base directory containing the UniVR folder.
                  If None, uses the parent directory of this file.
        model_type: Model type ('rife', 'softsplat', 'superslomo').
                   If specified, only adds paths for that model type.
                   If None, adds paths for all model types (legacy behavior).
    """
    if base_dir is None:
        # Resolve from package install location (pave_univr/)
        base_dir = os.path.dirname(__file__)

    univr_root = os.path.join(base_dir, 'UniVR')

    # Common paths that all models need
    paths_to_add = [
        univr_root,
        os.path.join(univr_root, 'package_core')
    ]

    # Add model-specific paths only
    model_type_lower = None
    if model_type:
        model_type_lower = model_type.lower()
        if model_type_lower == 'softsplat':
            softsplat_deep_unroll = os.path.join(univr_root, 'UniVR_SoftSplat', 'deep_unroll_net')
            paths_to_add.extend([
                softsplat_deep_unroll,
                os.path.join(softsplat_deep_unroll, 'softsplat_main')
            ])
        elif model_type_lower == 'superslomo':
            superslomo_deep_unroll = os.path.join(univr_root, 'UniVR_SuperSloMo', 'deep_unroll_net')
            paths_to_add.extend([
                superslomo_deep_unroll,
                os.path.join(superslomo_deep_unroll, 'superslomo')
            ])

    if model_type_lower is None or model_type_lower == "rife":
        rife_deep_unroll = os.path.join(univr_root, 'UniVR_RIFE', 'deep_unroll_net')
        paths_to_add.extend([
            rife_deep_unroll,
            os.path.join(rife_deep_unroll, 'RIFE')
        ])


    for path in paths_to_add:
        if path not in sys.path:
            sys.path.insert(0, path)


def validate_model_weights(log_dir: str) -> None:
    """Validate that model weights directory exists.

    Args:
        log_dir: Path to directory containing model weights

    Raises:
        FileNotFoundError: If weights directory doesn't exist or no weight files found
    """
    if not os.path.exists(log_dir):
        raise FileNotFoundError(f"Model weights directory not found: {log_dir}")

    weight_files = list(Path(log_dir).glob("*.pth")) + list(Path(log_dir).glob("*.pkl"))
    if not weight_files:
        raise FileNotFoundError(f"No model weight files found in: {log_dir}")
