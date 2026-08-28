import argparse
import os
import random
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Loader_data currently resolves Vector_data relative to the working directory.
os.chdir(PROJECT_ROOT)

from Loader_data import load_data
from model_CACL import CACL
from utils.TSNE import TSNE_plot


def parse_dataset_name(value):
    """Return the canonical dataset name accepted by Loader_data."""
    aliases = {"amg": "AMG", "mirage": "mirage"}
    normalized = str(value).strip().lower()
    if normalized not in aliases:
        raise argparse.ArgumentTypeError(
            "dataset must be either 'AMG' or 'mirage'"
        )
    return aliases[normalized]


def setup_seed(seed):
    """Use the same random-seed configuration as Train_CACL.py."""
    torch.manual_seed(seed + 1)
    torch.cuda.manual_seed_all(seed + 2)
    np.random.seed(seed + 3)
    random.seed(seed + 4)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate the best CACL checkpoint on the test split."
    )
    parser.add_argument(
        "--dataset",
        type=parse_dataset_name,
        default=None,
        metavar="{AMG,mirage}",
        help=(
            "Dataset to evaluate. If --checkpoint is omitted, its name also "
            "selects <dataset>_CACL_best.pth. Defaults to AMG when neither "
            "argument is provided."
        ),
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help=(
            "Optional path to a best-validation checkpoint. Its stored "
            "dataset name must match --dataset when both are supplied."
        ),
    )
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument(
        "--data_mode",
        choices=("coarse", "fine"),
        default="coarse",
        help=(
            "AMG label level: coarse uses binary y; fine uses six-class "
            "y_source. Mirage supports coarse only."
        ),
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="For example: cuda, cuda:0, or cpu. Defaults to CUDA when available.",
    )
    parser.add_argument(
        "--tsne_dir",
        type=Path,
        default=PROJECT_ROOT / "Classification_Figure" / "TSNE" / "test",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    requested_dataset = args.dataset
    if args.checkpoint is None:
        dataset_name = requested_dataset or "AMG"
        checkpoint_path = (
            Path(__file__).resolve().parent
            / f"{dataset_name}_CACL_best.pth"
        )
    else:
        checkpoint_path = args.checkpoint.resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}. Train the model first."
        )

    device = torch.device(
        args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=True,
    )
    test_seed = int(checkpoint.get("seed", 5))
    setup_seed(test_seed)

    checkpoint_dataset = checkpoint.get("dataset_name")
    if requested_dataset is None:
        dataset_name = (
            parse_dataset_name(checkpoint_dataset)
            if checkpoint_dataset is not None
            else "AMG"
        )
    else:
        dataset_name = requested_dataset
        if (
            checkpoint_dataset is not None
            and parse_dataset_name(checkpoint_dataset) != dataset_name
        ):
            raise ValueError(
                f"Dataset mismatch: --dataset={dataset_name}, but checkpoint "
                f"stores dataset_name={checkpoint_dataset}."
            )
    X, Y = load_data(dataset_name)
    test_views = X[2]
    if dataset_name == "AMG":
        label_index = 0 if args.data_mode == "coarse" else 1
        train_labels = np.asarray(Y[0][label_index]).reshape(-1)
        test_labels = np.asarray(Y[2][label_index]).reshape(-1)
    else:
        if args.data_mode != "coarse":
            raise ValueError("Mirage provides only coarse binary labels")
        train_labels = np.asarray(Y[0]).reshape(-1)
        test_labels = np.asarray(Y[2]).reshape(-1)
    train_labels = train_labels.astype(np.int64, copy=False)
    test_labels = test_labels.astype(np.int64, copy=False)

    view_num = int(checkpoint.get("view_num", len(test_views)))
    label_class_num = int(np.unique(train_labels).size)
    class_num = int(checkpoint.get("class_num", label_class_num))
    auto_dim = checkpoint.get("auto_dim")
    if auto_dim is None:
        raise KeyError("Checkpoint does not contain the model architecture: auto_dim")
    if len(test_views) != view_num:
        raise ValueError(
            f"Checkpoint expects {view_num} views, but test data has {len(test_views)}"
        )
    if class_num != label_class_num:
        raise ValueError(
            f"Checkpoint expects {class_num} classes, but "
            f"--data_mode={args.data_mode} provides {label_class_num}."
        )
    if any(view.shape[0] != test_labels.shape[0] for view in test_views):
        raise ValueError("Test feature/label sample counts are inconsistent")

    model_args = SimpleNamespace(model="Classification")
    model = CACL(
        auto_dim=auto_dim,
        device=device,
        view_num=view_num,
        cluster_n=class_num,
        args=model_args,
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    predictions = []
    tsne_features = []
    with torch.inference_mode():
        for start in range(0, test_labels.shape[0], args.batch_size):
            end = min(start + args.batch_size, test_labels.shape[0])
            data_list = [
                torch.as_tensor(
                    view[start:end],
                    dtype=torch.float32,
                    device=device,
                )
                for view in test_views
            ]
            *_, hlz, _, _, _, pseudo_list, _, _ = model(data_list)

            mean_probability = torch.stack(pseudo_list, dim=0).mean(dim=0)
            predictions.append(mean_probability.argmax(dim=1).cpu().numpy())

            mean_hlz = torch.stack(hlz, dim=0).mean(dim=0)
            tsne_features.append(mean_hlz.cpu().numpy())

    test_predictions = np.concatenate(predictions).astype(np.int64, copy=False)
    test_features = np.concatenate(tsne_features, axis=0)

    test_acc = accuracy_score(test_labels, test_predictions)
    test_precision = precision_score(
        test_labels, test_predictions, average="macro", zero_division=0
    )
    test_f1 = f1_score(
        test_labels, test_predictions, average="macro", zero_division=0
    )
    test_recall = recall_score(
        test_labels, test_predictions, average="macro", zero_division=0
    )

    print(f"Checkpoint: {checkpoint_path}")
    print(f"Train/Test seed: {test_seed}")
    print(
        f"Best validation: epoch={checkpoint.get('epoch', 'unknown')}, "
        f"valid_ACC={checkpoint.get('valid_acc', float('nan')):.4f}"
    )
    print(f"test_ACC: {test_acc:.4f}")
    print(f"test_PREC: {test_precision:.4f}")
    print(f"test_F1: {test_f1:.4f}")
    print(f"test_Recall: {test_recall:.4f}")

    TSNE_plot(
        test_features,
        test_labels,
        test_predictions,
        save_dir=args.tsne_dir,
        random_state=test_seed,
    )


if __name__ == "__main__":
    main()
