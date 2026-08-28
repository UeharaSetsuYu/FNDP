"""Export aligned AMG visual and text features for each official split.

Train, validation, and test remain separate. Images are encoded with frozen
DINOv2. Videos are uniformly sampled to eight frames and their DINOv2 frame
vectors are averaged. Text is encoded with the same frozen SBERT model used by
the existing MiRAGeNews preprocessing.
"""

from __future__ import annotations

import argparse
import gc
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from PIL import Image, ImageFile
from scipy.io import savemat
from tqdm.auto import tqdm

try:
    from .img2vec import FrozenDINOv2Encoder
    from .SBERT_model_init import HF_HUB_CACHE, download_sbert
except ImportError:  # Support running this file directly.
    from img2vec import FrozenDINOv2Encoder
    from SBERT_model_init import HF_HUB_CACHE, download_sbert


DATASET_ROOT = Path(
    r"D:\Data_Mining\Code\Experiment\FakeNewsDetectionByMVL\dataset\AMG"
)
OUTPUT_DIR = Path(
    r"D:\Data_Mining\Code\Experiment\FakeNewsDetectionByMVL\Vector_data"
)
MODEL_CACHE_DIR = HF_HUB_CACHE
SPLITS = ("train", "val", "test")
OUTPUT_SPLIT_NAMES = {"train": "train", "val": "valid", "test": "test"}
MEDIA_SUFFIXES = {".jpg", ".jpeg", ".png", ".mp4"}
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}
SOURCE_LABELS = set(range(6))

# These 13 released AMG image files contain only the text
# "URL signature expired" instead of valid image bytes.
UNUSABLE_IMAGE_IDS = frozenset(
    {
        "2095",
        "2115",
        "2120",
        "2356",
        "2417",
        "2899",
        "3035",
        "3983",
        "4606",
        "4628",
        "4641",
        "4707",
        "4815",
    }
)

# Additional train media verified as unusable: one 21-byte MP4 that cannot be
# opened, and two 1x1 placeholder PNGs with no meaningful visual content.
UNUSABLE_TRAIN_MEDIA_IDS = frozenset({"2759", "2966", "3148"})
UNUSABLE_MEDIA_IDS = UNUSABLE_IMAGE_IDS | UNUSABLE_TRAIN_MEDIA_IDS


@dataclass(frozen=True)
class AMGRecord:
    """One cleaned AMG sample."""

    split: str
    sample_id: str
    text: str
    source_label: int
    media_path: Path

    @property
    def binary_label(self) -> int:
        """Return 0 for real news and 1 for any fake-news source."""
        return int(self.source_label != 0)


def _read_annotations(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as file:
        rows = json.load(file)
    if not isinstance(rows, list):
        raise TypeError(f"Expected a JSON list in {path}")
    return rows


def _index_media(split_dir: Path) -> dict[str, Path]:
    if not split_dir.is_dir():
        raise FileNotFoundError(f"Media split directory not found: {split_dir}")

    media_by_id: dict[str, Path] = {}
    for path in split_dir.iterdir():
        if not path.is_file() or path.suffix.lower() not in MEDIA_SUFFIXES:
            continue
        if path.stem in media_by_id:
            raise ValueError(
                f"Duplicate media ID '{path.stem}' in {split_dir}"
            )
        media_by_id[path.stem] = path
    return media_by_id


def load_amg_records(
    dataset_root: str | Path = DATASET_ROOT,
    *,
    splits: Sequence[str] = SPLITS,
) -> list[AMGRecord]:
    """Load and clean AMG while preserving split and JSON order."""
    dataset_root = Path(dataset_root)
    if len(UNUSABLE_IMAGE_IDS) != 13:
        raise RuntimeError("The AMG cleaning list must contain 13 sample IDs")
    if len(UNUSABLE_TRAIN_MEDIA_IDS) != 3:
        raise RuntimeError(
            "The AMG train-media cleaning list must contain 3 sample IDs"
        )

    records: list[AMGRecord] = []
    seen_ids: set[str] = set()
    removed_ids: list[str] = []

    for split in splits:
        annotation_path = dataset_root / "annotations" / f"{split}.json"
        if not annotation_path.is_file():
            raise FileNotFoundError(
                f"Annotation file not found: {annotation_path}"
            )
        media_by_id = _index_media(dataset_root / "AMG_MEDIA" / split)
        split_records = 0

        for row in _read_annotations(annotation_path):
            sample_id = str(row["Id"])
            if sample_id in UNUSABLE_MEDIA_IDS:
                removed_ids.append(sample_id)
                continue
            if sample_id in seen_ids:
                raise ValueError(f"Duplicate annotation ID: {sample_id}")

            media_path = media_by_id.get(sample_id)
            if media_path is None:
                raise FileNotFoundError(
                    f"No media file found for {split} sample {sample_id}"
                )

            text = row["content"]
            if not isinstance(text, str):
                raise TypeError(
                    f"Sample {sample_id} has a non-string content field"
                )
            source_label = int(row["label"])
            if source_label not in SOURCE_LABELS:
                raise ValueError(
                    f"Sample {sample_id} has invalid source label "
                    f"{source_label}; expected 0-5"
                )

            records.append(
                AMGRecord(
                    split=split,
                    sample_id=sample_id,
                    text=text,
                    source_label=source_label,
                    media_path=media_path,
                )
            )
            seen_ids.add(sample_id)
            split_records += 1

        print(f"Loaded {split}: {split_records} cleaned samples")

    if not records:
        raise ValueError("No AMG records were loaded")

    print(
        "Removed unusable samples present in selected splits: "
        f"{len(removed_ids)} ({', '.join(removed_ids)})"
    )
    return records


def _sample_video_frames(
    video_path: Path,
    *,
    frame_count: int = 8,
) -> list[Image.Image]:
    """Uniformly sample RGB frames over the decodable video duration."""
    if frame_count <= 0:
        raise ValueError("frame_count must be greater than zero")

    try:
        import cv2
    except ImportError as error:
        raise ImportError(
            "OpenCV is required for AMG video processing. "
            "Install it with: pip install opencv-python"
        ) from error

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        capture.release()
        raise OSError(f"Could not open video: {video_path}")

    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        capture.release()
        raise OSError(f"Video reports no decodable frames: {video_path}")

    # Fast path: random access works for most AMG videos.
    # Repeated indices are intentional for videos shorter than frame_count.
    indices = np.rint(
        np.linspace(0, total_frames - 1, num=frame_count)
    ).astype(np.int64)
    frames: list[Image.Image] = []
    failed_index: int | None = None
    try:
        for index in indices:
            capture.set(cv2.CAP_PROP_POS_FRAMES, int(index))
            ok, frame = capture.read()
            if not ok or frame is None:
                failed_index = int(index)
                break
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(Image.fromarray(rgb))
    finally:
        capture.release()

    if failed_index is None:
        return frames

    for frame in frames:
        frame.close()
    print(
        "Random video seek failed; retrying with sequential decoding: "
        f"{video_path} (frame {failed_index})"
    )

    # Some AMG MP4 files have unreliable random-seek indices or report more
    # frames than are actually decodable. Count the real frames sequentially,
    # then sample again from that verified range.
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        capture.release()
        raise OSError(f"Could not reopen video: {video_path}")
    decodable_frames = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok or frame is None:
                break
            decodable_frames += 1
    finally:
        capture.release()

    if decodable_frames <= 0:
        raise OSError(f"Video contains no decodable frames: {video_path}")

    indices = np.rint(
        np.linspace(0, decodable_frames - 1, num=frame_count)
    ).astype(np.int64)
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        capture.release()
        raise OSError(f"Could not reopen video: {video_path}")

    frames = []
    target_position = 0
    try:
        for frame_index in range(decodable_frames):
            ok, frame = capture.read()
            if not ok or frame is None:
                break
            while (
                target_position < len(indices)
                and int(indices[target_position]) == frame_index
            ):
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(Image.fromarray(rgb.copy()))
                target_position += 1
            if target_position == len(indices):
                break
    finally:
        capture.release()

    if len(frames) != frame_count:
        for frame in frames:
            frame.close()
        raise OSError(
            f"Sequential decoding returned {len(frames)}/{frame_count} "
            f"sampled frames from {video_path}"
        )
    return frames


def _load_visual_inputs(
    record: AMGRecord,
    *,
    video_frame_count: int,
) -> list[Image.Image]:
    suffix = record.media_path.suffix.lower()
    if suffix in IMAGE_SUFFIXES:
        try:
            with Image.open(record.media_path) as image:
                return [image.convert("RGB")]
        except OSError as error:
            if "image file is truncated" not in str(error).lower():
                raise OSError(
                    f"Could not decode image for sample {record.sample_id}: "
                    f"{record.media_path}"
                ) from error

            # AMG sample 4362 is a usable JPEG whose end marker is slightly
            # truncated. Retry only this specific Pillow error in permissive
            # mode, and restore the process-wide setting immediately after.
            previous_setting = ImageFile.LOAD_TRUNCATED_IMAGES
            ImageFile.LOAD_TRUNCATED_IMAGES = True
            try:
                with Image.open(record.media_path) as image:
                    recovered = image.convert("RGB")
            except Exception as retry_error:
                raise OSError(
                    f"Could not recover truncated image for sample "
                    f"{record.sample_id}: {record.media_path}"
                ) from retry_error
            finally:
                ImageFile.LOAD_TRUNCATED_IMAGES = previous_setting
            print(
                f"Recovered truncated image for sample {record.sample_id}: "
                f"{record.media_path}"
            )
            return [recovered]
        except Exception as error:
            raise OSError(
                f"Could not decode image for sample {record.sample_id}: "
                f"{record.media_path}"
            ) from error
    if suffix == ".mp4":
        return _sample_video_frames(
            record.media_path,
            frame_count=video_frame_count,
        )
    raise ValueError(f"Unsupported media type: {record.media_path}")


def _label_arrays(records: Sequence[AMGRecord]) -> tuple[np.ndarray, np.ndarray]:
    y_source = np.asarray(
        [record.source_label for record in records], dtype=np.int64
    ).reshape(-1, 1)
    y = np.asarray(
        [record.binary_label for record in records], dtype=np.int64
    ).reshape(-1, 1)
    return y_source, y


def encode_visual_records(
    records: Sequence[AMGRecord],
    encoder: FrozenDINOv2Encoder,
    *,
    record_batch_size: int = 16,
    dinov2_batch_size: int = 32,
    video_frame_count: int = 8,
) -> np.ndarray:
    """Encode images and mean-pooled video frames in record order."""
    if record_batch_size <= 0 or dinov2_batch_size <= 0:
        raise ValueError("Batch sizes must be greater than zero")

    feature_blocks: list[np.ndarray] = []
    progress = tqdm(
        total=len(records),
        desc=f"DINOv2 {records[0].split if records else 'empty'}",
        unit="sample",
    )

    for start in range(0, len(records), record_batch_size):
        record_batch = records[start : start + record_batch_size]
        frame_inputs: list[Image.Image] = []
        spans: list[tuple[int, int]] = []

        try:
            for record in record_batch:
                sample_frames = _load_visual_inputs(
                    record,
                    video_frame_count=video_frame_count,
                )
                span_start = len(frame_inputs)
                frame_inputs.extend(sample_frames)
                spans.append((span_start, len(frame_inputs)))

            frame_vectors = encoder.encode(
                frame_inputs,
                batch_size=dinov2_batch_size,
                return_cpu=True,
            )
            sample_vectors = torch.stack(
                [frame_vectors[left:right].mean(dim=0) for left, right in spans],
                dim=0,
            )
            feature_blocks.append(
                sample_vectors.numpy().astype(np.float32, copy=False)
            )
        finally:
            for image in frame_inputs:
                image.close()

        progress.update(len(record_batch))

    progress.close()
    visual_vectors = np.concatenate(feature_blocks, axis=0)
    if visual_vectors.shape[0] != len(records):
        raise RuntimeError(
            "Visual alignment failed: "
            f"features={visual_vectors.shape[0]}, records={len(records)}"
        )
    return visual_vectors


def encode_text_records(
    records: Sequence[AMGRecord],
    model: Any,
    *,
    batch_size: int = 64,
) -> np.ndarray:
    """Encode AMG content fields with frozen SBERT in record order."""
    if batch_size <= 0:
        raise ValueError("batch_size must be greater than zero")
    vectors = model.encode(
        [record.text for record in records],
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=False,
    )
    text_vectors = np.asarray(vectors, dtype=np.float32)
    if text_vectors.ndim != 2 or text_vectors.shape[0] != len(records):
        raise RuntimeError(
            "Text alignment failed: "
            f"features={text_vectors.shape}, records={len(records)}"
        )
    return text_vectors


def _save_mat(
    output_path: Path,
    features: np.ndarray,
    y_source: np.ndarray,
    y: np.ndarray,
) -> None:
    if not (
        features.shape[0] == y_source.shape[0] == y.shape[0]
    ):
        raise RuntimeError(
            "MAT alignment failed: "
            f"X={features.shape}, y_source={y_source.shape}, y={y.shape}"
        )

    temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
    savemat(
        temporary_path,
        {"X": features, "y_source": y_source, "y": y},
        appendmat=False,
        do_compression=True,
    )
    temporary_path.replace(output_path)
    print(
        f"Saved X={features.shape}, y_source={y_source.shape}, "
        f"y={y.shape} -> {output_path}"
    )


def convert_amg_splits_to_mat(
    *,
    dataset_root: str | Path = DATASET_ROOT,
    output_dir: str | Path = OUTPUT_DIR,
    model_cache_dir: str | None = MODEL_CACHE_DIR,
    device: str | torch.device | None = None,
    record_batch_size: int = 16,
    dinov2_batch_size: int = 32,
    text_batch_size: int = 64,
    video_frame_count: int = 8,
    overwrite: bool = False,
) -> dict[str, Path]:
    """Export aligned image/text MAT files for cleaned AMG splits."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        f"img_{split}": output_dir
        / f"AMG_img_{OUTPUT_SPLIT_NAMES[split]}.mat"
        for split in SPLITS
    }
    outputs.update(
        {
            f"text_{split}": output_dir
            / f"AMG_text_{OUTPUT_SPLIT_NAMES[split]}.mat"
            for split in SPLITS
        }
    )
    if not overwrite:
        existing = [path for path in outputs.values() if path.exists()]
        if existing:
            raise FileExistsError(
                "Output already exists: "
                + ", ".join(str(path) for path in existing)
                + ". Use --overwrite to replace it."
            )

    records = load_amg_records(dataset_root, splits=SPLITS)
    records_by_split = {
        split: [record for record in records if record.split == split]
        for split in SPLITS
    }

    dino_encoder = FrozenDINOv2Encoder(
        cache_dir=model_cache_dir,
        device=device,
        pooling="cls",
        normalize=False,
    )
    for split in SPLITS:
        split_records = records_by_split[split]
        y_source, y = _label_arrays(split_records)
        visual_vectors = encode_visual_records(
            split_records,
            dino_encoder,
            record_batch_size=record_batch_size,
            dinov2_batch_size=dinov2_batch_size,
            video_frame_count=video_frame_count,
        )
        _save_mat(outputs[f"img_{split}"], visual_vectors, y_source, y)
        del visual_vectors

    # Release DINOv2 before SBERT is loaded to keep peak GPU memory small.
    del dino_encoder
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    sbert_model = download_sbert(
        cache_dir=model_cache_dir,
        device=str(device) if device is not None else None,
    )
    for split in SPLITS:
        split_records = records_by_split[split]
        y_source, y = _label_arrays(split_records)
        text_vectors = encode_text_records(
            split_records,
            sbert_model,
            batch_size=text_batch_size,
        )
        _save_mat(outputs[f"text_{split}"], text_vectors, y_source, y)
        del text_vectors

    if any(path.stat().st_size == 0 for path in outputs.values()):
        raise RuntimeError("One or more MAT outputs are empty")
    return outputs


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Encode cleaned AMG train/valid/test visual and text features"
    )
    parser.add_argument("--dataset-root", type=Path, default=DATASET_ROOT)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--device", default=None)
    parser.add_argument("--record-batch-size", type=int, default=16)
    parser.add_argument("--dinov2-batch-size", type=int, default=32)
    parser.add_argument("--text-batch-size", type=int, default=64)
    parser.add_argument("--video-frames", type=int, default=8)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    convert_amg_splits_to_mat(
        dataset_root=args.dataset_root,
        output_dir=args.output_dir,
        device=args.device,
        record_batch_size=args.record_batch_size,
        dinov2_batch_size=args.dinov2_batch_size,
        text_batch_size=args.text_batch_size,
        video_frame_count=args.video_frames,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
