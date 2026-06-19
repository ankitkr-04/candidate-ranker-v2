"""Ensure the Qwen3-4B SLM is available locally, downloading it on first use.

This runs only in the precompute stage, where network access is permitted. The
model lives under assets/model/ (gitignored) and is fetched once; later calls
are no-ops. The huggingface_hub import is deferred so this module remains
importable in the ranking environment, which does not install it.
"""

from __future__ import annotations

import os
from pathlib import Path

from src.paths import MODEL_DIR

MODEL_REPO = "Qwen/Qwen3-4B-Instruct-2507"

# Files required to load the model for inference. Repo metadata such as
# README/LICENSE is skipped so the download stays limited to what we use.
ALLOW_PATTERNS = [
    "config.json",
    "generation_config.json",
    "model.safetensors.index.json",
    "model-*.safetensors",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
    "merges.txt",
]

_REQUIRED_FILES = ("config.json", "model.safetensors.index.json", "tokenizer.json")


def is_model_present(local_dir: Path = MODEL_DIR) -> bool:
    """Report whether a usable model already exists at local_dir."""
    has_required = all((local_dir / name).is_file() for name in _REQUIRED_FILES)
    has_weights = any(local_dir.glob("model-*.safetensors"))
    return has_required and has_weights


def ensure_model(local_dir: Path = MODEL_DIR) -> Path:
    """Return the local model path, downloading from Hugging Face if absent."""
    if is_model_present(local_dir):
        return local_dir

    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
    from huggingface_hub import snapshot_download  # type: ignore[import-not-found]

    local_dir.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=MODEL_REPO,
        local_dir=str(local_dir),
        allow_patterns=ALLOW_PATTERNS,
    )
    if not is_model_present(local_dir):
        raise RuntimeError(f"Model download to {local_dir} did not produce the expected files")
    return local_dir


if __name__ == "__main__":
    print(f"Model ready at: {ensure_model()}")
