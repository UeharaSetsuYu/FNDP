"""Download and load a frozen DINOv2 image encoder."""

from __future__ import annotations

import os
from typing import Any


MODEL_NAME = "facebook/dinov2-base"
DEFAULT_PROXY = os.getenv("HF_PROXY", "http://127.0.0.1:7897")
HF_HOME = os.getenv("HF_HOME", r"D:\HFCache")
HF_HUB_CACHE = os.getenv("HF_HUB_CACHE", os.path.join(HF_HOME, "hub"))
os.environ.setdefault("HF_HOME", HF_HOME)
os.environ.setdefault("HF_HUB_CACHE", HF_HUB_CACHE)


def download_dinov2(
    *,
    model_name: str = MODEL_NAME,
    cache_dir: str | None = HF_HUB_CACHE,
    proxy: str | None = DEFAULT_PROXY,
) -> tuple[Any, Any]:
    """Download DINOv2 to the Hugging Face cache and return processor/model."""
    if proxy:
        os.environ["HTTP_PROXY"] = proxy
        os.environ["HTTPS_PROXY"] = proxy

    # Import after configuring the proxy so Hugging Face downloads use it.
    from transformers import AutoImageProcessor, AutoModel

    processor = AutoImageProcessor.from_pretrained(
        model_name,
        cache_dir=cache_dir,
    )
    model = AutoModel.from_pretrained(
        model_name,
        cache_dir=cache_dir,
    )
    model.requires_grad_(False)
    model.eval()

    return processor, model





if __name__ == "__main__":
    download_dinov2()
    print(f"DINOv2 model is ready: {MODEL_NAME}")

