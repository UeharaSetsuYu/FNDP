"""Extract image features with a frozen DINOv2 encoder."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Literal, Union

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from scipy.io import savemat
from torch import Tensor
from tqdm.auto import tqdm

try:
    from data_loader import DATASET_NAME, load_mirage_news
except ModuleNotFoundError:  # Support running this file directly.
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from data_loader import DATASET_NAME, load_mirage_news


try:
    from .DINOv2_model_init import DEFAULT_PROXY, MODEL_NAME, download_dinov2
except ImportError:  # Support running this file directly.
    from DINOv2_model_init import DEFAULT_PROXY, MODEL_NAME, download_dinov2


ImageInput = Union[Image.Image, str, Path]
Pooling = Literal["cls", "mean", "cls_mean"]
OUTPUT_DIR = Path(
    r"D:\Data_Mining\Code\Experiment\FakeNewsDetectionByMVL\Vector_data"
)
MODEL_CACHE_DIR = r"D:\HFCache\hub"
SPLITS = ("train", "valid", "test")


class FrozenDINOv2Encoder:
    """Reusable frozen DINOv2 image feature extractor."""

    def __init__(
        self,
        *,
        model_name: str = MODEL_NAME,
        cache_dir: str | None = None,
        proxy: str | None = DEFAULT_PROXY,
        device: str | torch.device | None = None,
        pooling: Pooling = "cls",
        normalize: bool = False,
    ) -> None:
        if pooling not in {"cls", "mean", "cls_mean"}:
            raise ValueError(
                "pooling must be one of: 'cls', 'mean', 'cls_mean'"
            )

        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.pooling = pooling
        self.normalize = normalize
        self.processor, self.model = download_dinov2(
            model_name=model_name,
            cache_dir=cache_dir,
            proxy=proxy,
        )
        self.model.requires_grad_(False)
        self.model.eval()
        self.model.to(self.device)

    @torch.inference_mode()
    def encode(
        self,
        images: ImageInput | Sequence[ImageInput],
        *,
        batch_size: int = 32,
        return_cpu: bool = True,
    ) -> Tensor:
        """Return one feature vector per image with shape ``[N, D]``."""
        if batch_size <= 0:
            raise ValueError("batch_size must be greater than zero")

        image_list = self._prepare_images(images)
        features: list[Tensor] = []
        self.model.eval()

        for start in range(0, len(image_list), batch_size):
            batch = image_list[start : start + batch_size]
            inputs = self.processor(
                images=batch,
                return_tensors="pt",
            ).to(self.device)
            tokens = self.model(**inputs).last_hidden_state
            feature = self._pool(tokens)

            if self.normalize:
                feature = F.normalize(feature, dim=-1)
            if return_cpu:
                feature = feature.cpu()

            features.append(feature)

        return torch.cat(features, dim=0)

    def _pool(self, tokens: Tensor) -> Tensor:
        cls_feature = tokens[:, 0, :]
        patch_mean = tokens[:, 1:, :].mean(dim=1)

        if self.pooling == "cls":
            return cls_feature
        if self.pooling == "mean":
            return patch_mean
        return torch.cat((cls_feature, patch_mean), dim=-1)

    @staticmethod
    def _prepare_images(
        images: ImageInput | Sequence[ImageInput],
    ) -> list[Image.Image]:
        if isinstance(images, (Image.Image, str, Path)):
            images = [images]

        prepared: list[Image.Image] = []
        for image in images:
            if isinstance(image, Image.Image):
                prepared.append(image.convert("RGB"))
            elif isinstance(image, (str, Path)):
                with Image.open(image) as opened:
                    prepared.append(opened.convert("RGB"))
            else:
                raise TypeError(
                    "Each image must be a PIL image or a filesystem path"
                )

        if not prepared:
            raise ValueError("At least one image is required")
        return prepared


def _dataset_file_name(split: str) -> str:
    dataset_name = DATASET_NAME.rsplit("/", maxsplit=1)[-1]
    dataset_name = dataset_name.replace("-", "_")
    return f"{dataset_name}_img_{split}.mat"


def _split_size(dataset: object) -> int | None:
    try:
        return len(dataset)  # type: ignore[arg-type]
    except TypeError:
        return None


def convert_split_to_mat(
    dataset: object,
    split: str,
    encoder: FrozenDINOv2Encoder,
    *,
    output_dir: str | Path = OUTPUT_DIR,
    batch_size: int = 32,
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
    batch_images: list[ImageInput] = []
    batch_labels: list[int] = []

    def encode_batch() -> None:
        if not batch_images:
            return
        batch_features = encoder.encode(
            batch_images,
            batch_size=len(batch_images),
            return_cpu=True,
        )
        feature_blocks.append(
            batch_features.numpy().astype(np.float32, copy=False)
        )
        labels.extend(batch_labels)
        batch_images.clear()
        batch_labels.clear()

    progress = tqdm(
        dataset,  # type: ignore[arg-type]
        total=_split_size(dataset),
        desc=f"Encoding {split}",
        unit="image",
    )
    for sample in progress:
        batch_images.append(sample["image"])
        batch_labels.append(int(sample["label"]))
        if len(batch_images) == batch_size:
            encode_batch()
    encode_batch()

    if not feature_blocks:
        raise ValueError(f"Split '{split}' contains no samples")

    image_vectors = np.concatenate(feature_blocks, axis=0)
    label_vector = np.asarray(labels, dtype=np.int64).reshape(-1, 1)
    if image_vectors.shape[0] != label_vector.shape[0]:
        raise RuntimeError(
            "Image/label alignment failed: "
            f"X has {image_vectors.shape[0]} rows but "
            f"Y has {label_vector.shape[0]} rows"
        )

    temporary_path = output_path.with_suffix(".tmp")
    savemat(
        temporary_path,
        {"X": image_vectors, "Y": label_vector},
        appendmat=False,
    )
    temporary_path.replace(output_path)
    print(
        f"Saved {split}: X={image_vectors.shape}, "
        f"Y={label_vector.shape} -> {output_path}"
    )
    return output_path


def convert_mirage_news_to_mat(
    *,
    output_dir: str | Path = OUTPUT_DIR,
    model_cache_dir: str | None = MODEL_CACHE_DIR,
    batch_size: int = 32,
    device: str | torch.device | None = None,
    overwrite: bool = False,
    streaming: bool = True,
) -> dict[str, Path]:
    """Encode MiRAGeNews train/valid/test splits into separate MAT files."""
    datasets = load_mirage_news(streaming=streaming)
    encoder = FrozenDINOv2Encoder(
        cache_dir=model_cache_dir,
        device=device,
        pooling="cls",
    )
    return {
        split: convert_split_to_mat(
            datasets[split],
            split,
            encoder,
            output_dir=output_dir,
            batch_size=batch_size,
            overwrite=overwrite,
        )
        for split in SPLITS
    }


def main() -> None:
    convert_mirage_news_to_mat()


if __name__ == "__main__":
    main()
