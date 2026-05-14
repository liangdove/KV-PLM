import argparse
import os
import sys
from typing import List, Tuple

import numpy as np
import pandas as pd


def _add_chemvl_private_to_path(chemvl_private_root: str) -> None:
    chemvl_private_root = os.path.abspath(chemvl_private_root)
    if chemvl_private_root not in sys.path:
        sys.path.insert(0, chemvl_private_root)


def _task_to_kv_name(task: str) -> str:
    lower = task.strip().lower()
    if lower in ["tox21", "sider"]:
        return lower
    return task.strip().upper()


def _task_to_chemvl_dir(task: str) -> str:
    return task.strip().lower()


def _parse_labels(label_series: pd.Series) -> np.ndarray:
    labels: List[List[int]] = []
    is_multitask = False
    for val in label_series:
        parts = str(val).strip().split(" ")
        parts = [p for p in parts if p != ""]
        if len(parts) > 1:
            is_multitask = True
            labels.append([int(p) for p in parts])
        else:
            labels.append([int(parts[0])])
    arr = np.array(labels, dtype=int)
    if is_multitask:
        return arr
    return arr.reshape(-1)


def _load_chemvl_csv(chemvl_root: str, task: str) -> Tuple[List[str], np.ndarray]:
    dataset_dir = _task_to_chemvl_dir(task)
    csv_path = os.path.join(
        chemvl_root,
        dataset_dir,
        "processed",
        f"{dataset_dir}_processed_ac.csv",
    )
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"ChemVL CSV not found: {csv_path}")
    df = pd.read_csv(csv_path)
    if "smiles" not in df.columns or "label" not in df.columns:
        raise ValueError("ChemVL CSV must contain smiles and label columns.")
    smiles = df["smiles"].astype(str).tolist()
    labels = _parse_labels(df["label"])
    return smiles, labels


def _save_kv_files(out_dir: str, task: str, smiles: List[str], labels: np.ndarray) -> None:
    os.makedirs(out_dir, exist_ok=True)
    kv_name = _task_to_kv_name(task)
    sm_path = os.path.join(out_dir, f"sm_{kv_name}.npy")
    lab_path = os.path.join(out_dir, f"lab_{kv_name}.npy")
    text_path = os.path.join(out_dir, f"text_{kv_name}.txt")

    np.save(sm_path, np.asarray(smiles, dtype=str))
    np.save(lab_path, labels)
    with open(text_path, "w", encoding="utf-8") as f:
        for smi in smiles:
            f.write(f"{smi}\n")


def _split_indices(
    split_mode: str,
    smiles: List[str],
    seed: int,
    include_chirality: bool,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    split_mode = split_mode.strip().lower()
    idx = list(range(len(smiles)))
    if split_mode == "scaffold":
        from utils.splitter import scaffold_split_train_val_test

        return scaffold_split_train_val_test(
            idx,
            smiles,
            frac_train=0.8,
            frac_valid=0.1,
            frac_test=0.1,
            include_chirality=include_chirality,
        )
    if split_mode == "random_scaffold":
        from utils.splitter import random_scaffold_split_train_val_test

        return random_scaffold_split_train_val_test(
            idx,
            smiles,
            frac_train=0.8,
            frac_valid=0.1,
            frac_test=0.1,
            seed=seed,
            include_chirality=include_chirality,
        )
    raise ValueError(f"Unsupported split_mode: {split_mode}")


def _save_split_indices(out_dir: str, task: str, split_mode: str, splits) -> None:
    kv_name = _task_to_kv_name(task)
    split_mode = split_mode.strip().lower()
    train_idx, val_idx, test_idx = splits
    np.save(os.path.join(out_dir, f"split_{kv_name}_{split_mode}_train.npy"), np.asarray(train_idx))
    np.save(os.path.join(out_dir, f"split_{kv_name}_{split_mode}_valid.npy"), np.asarray(val_idx))
    np.save(os.path.join(out_dir, f"split_{kv_name}_{split_mode}_test.npy"), np.asarray(test_idx))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True, type=str, help="BBBP, HIV, tox21, sider")
    parser.add_argument(
        "--chemvl_root",
        default="/home/administrator/code_liangdove/chemvl-data/finetuning_datasets/MPP/classification",
        type=str,
    )
    parser.add_argument(
        "--chemvl_private_root",
        default="/home/administrator/code_liangdove/ChemVL-private",
        type=str,
    )
    parser.add_argument(
        "--out_dir",
        default=None,
        type=str,
        help="KV-PLM/MoleculeNet output dir",
    )
    parser.add_argument("--split_mode", default="scaffold", choices=["scaffold", "random_scaffold"])
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--include_chirality", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    out_dir = args.out_dir or os.path.join(project_root, "MoleculeNet")

    _add_chemvl_private_to_path(args.chemvl_private_root)

    smiles, labels = _load_chemvl_csv(args.chemvl_root, args.task)
    _save_kv_files(out_dir, args.task, smiles, labels)

    splits = _split_indices(args.split_mode, smiles, args.seed, args.include_chirality)
    _save_split_indices(out_dir, args.task, args.split_mode, splits)

    print("Done. KV-PLM MoleculeNet files written to:")
    print(out_dir)


if __name__ == "__main__":
    main()
