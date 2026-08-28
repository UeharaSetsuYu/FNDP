"""Load frozen SBERT and export MiRAGeNews text features to MAT files."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np
from scipy.io import savemat
from tqdm.auto import tqdm


MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"
DEFAULT_PROXY = os.getenv("HF_PROXY", "http://127.0.0.1:7897")
HF_HOME = os.getenv("HF_HOME", r"D:\HFCache")
HF_HUB_CACHE = os.getenv("HF_HUB_CACHE", os.path.join(HF_HOME, "hub"))

# This project uses the PyTorch implementation of Transformers. Explicitly
# disable its TensorFlow backend to avoid conflicts with Keras 3/tf-keras.
os.environ["USE_TF"] = "0"
os.environ["USE_TORCH"] = "1"
os.environ.setdefault("HF_HOME", HF_HOME)
os.environ.setdefault("HF_HUB_CACHE", HF_HUB_CACHE)

try:
    from data_loader import DATASET_NAME, load_mirage_news
except ModuleNotFoundError:  # Support running this file directly.
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from data_loader import DATASET_NAME, load_mirage_news


OUTPUT_DIR = Path(
    r"D:\Data_Mining\Code\Experiment\FakeNewsDetectionByMVL\Vector_data"
)
MODEL_CACHE_DIR = HF_HUB_CACHE
SPLITS = ("train", "valid", "test")


def download_sbert(
    *,
    model_name: str = MODEL_NAME,
    cache_dir: str | None = HF_HUB_CACHE,
    proxy: str | None = DEFAULT_PROXY,
    device: str | None = None,
) -> Any:
    """Download SBERT to the Hugging Face cache and return a frozen model."""
    if proxy:
        os.environ["HTTP_PROXY"] = proxy
        os.environ["HTTPS_PROXY"] = proxy

    # Import after configuring the backend, cache and proxy environment.
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(
        model_name,
        cache_folder=cache_dir,
        device=device,
    )
    model.requires_grad_(False)
    model.eval()

    return model


def _dataset_file_name(split: str) -> str:
    dataset_name = DATASET_NAME.rsplit("/", maxsplit=1)[-1]
    dataset_name = dataset_name.replace("-", "_")
    return f"{dataset_name}_text_{split}.mat"


def _split_size(dataset: object) -> int | None:
    try:
        return len(dataset)  # type: ignore[arg-type]
    except TypeError:
        return None


def convert_split_to_mat(
    dataset: object,
    split: str,
    model: Any,
    *,
    output_dir: str | Path = OUTPUT_DIR,
    batch_size: int = 64,
    normalize_embeddings: bool = False,
    overwrite: bool = False,
) -> Path:
    """Encode one split in iteration order and save aligned ``X``/``Y``."""
    if split not in SPLITS:
        raise ValueError(f"Unknown split: {split}")
    if batch_size <= 0:
        raise ValueError("batch_size must be greater than zero")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / _dataset_file_name(split)
    if output_path.exists() and not overwrite:
        raise FileExistsError(
            f"Output already exists: {output_path}. "
            "Use overwrite=True to replace it."
        )

    feature_blocks: list[np.ndarray] = []
    labels: list[int] = []
    batch_texts: list[str] = []
    batch_labels: list[int] = []

    def encode_batch() -> None:
        if not batch_texts:
            return

        batch_features = model.encode(
            batch_texts,
            batch_size=len(batch_texts),
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=normalize_embeddings,
        )
        batch_array = np.asarray(batch_features, dtype=np.float32)
        if batch_array.ndim != 2 or batch_array.shape[0] != len(batch_labels):
            raise RuntimeError(
                "Text encoding returned an invalid shape: "
                f"features={batch_array.shape}, labels={len(batch_labels)}"
            )

        feature_blocks.append(batch_array)
        labels.extend(batch_labels)
        batch_texts.clear()
        batch_labels.clear()

    progress = tqdm(
        dataset,  # type: ignore[arg-type]
        total=_split_size(dataset),
        desc=f"Encoding {split}",
        unit="text",
    )
    for sample in progress:
        text = sample["text"]
        if not isinstance(text, str):
            raise TypeError(
                f"Split '{split}' contains a non-string text value: "
                f"{type(text).__name__}"
            )
        batch_texts.append(text)
        batch_labels.append(int(sample["label"]))
        if len(batch_texts) == batch_size:
            encode_batch()
    encode_batch()

    if not feature_blocks:
        raise ValueError(f"Split '{split}' contains no samples")

    text_vectors = np.concatenate(feature_blocks, axis=0)
    label_vector = np.asarray(labels, dtype=np.int64).reshape(-1, 1)
    if text_vectors.shape[0] != label_vector.shape[0]:
        raise RuntimeError(
            "Text/label alignment failed: "
            f"X has {text_vectors.shape[0]} rows but "
            f"Y has {label_vector.shape[0]} rows"
        )

    temporary_path = output_path.with_suffix(".tmp")
    savemat(
        temporary_path,
        {"X": text_vectors, "Y": label_vector},
        appendmat=False,
    )
    temporary_path.replace(output_path)
    print(
        f"Saved {split}: X={text_vectors.shape}, "
        f"Y={label_vector.shape} -> {output_path}"
    )
    return output_path


def convert_mirage_news_to_mat(
    *,
    output_dir: str | Path = OUTPUT_DIR,
    model_cache_dir: str | None = MODEL_CACHE_DIR,
    batch_size: int = 64,
    device: str | None = None,
    normalize_embeddings: bool = False,
    overwrite: bool = False,
    streaming: bool = True,
) -> dict[str, Path]:
    """Encode MiRAGeNews train/valid/test text into separate MAT files."""
    datasets = load_mirage_news(streaming=streaming)
    model = download_sbert(
        cache_dir=model_cache_dir,
        device=device,
    )
    return {
        split: convert_split_to_mat(
            datasets[split],
            split,
            model,
            output_dir=output_dir,
            batch_size=batch_size,
            normalize_embeddings=normalize_embeddings,
            overwrite=overwrite,
        )
        for split in SPLITS
    }


def main() -> None:
    convert_mirage_news_to_mat()


if __name__ == "__main__":
    main()
