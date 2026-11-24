# data/split_manifests

The official split (80/10/10, chosen by the split-ratio experiment in notebook Cell 5.3). **Every model and every member uses these files.**

| File | Images |
|---|---|
| `train_split.csv` | 240 (80 per class) |
| `validation_split.csv` | 30 (10 per class) |
| `test_split.csv` | 30 (10 per class) |
| `data_summary.json` | Counts and leakage-check results (written by notebook Cell 5.4 or `scripts/prepare_data.py`) |

Columns: `filename, label, label_id, split, relative_path`. The paths are relative,
so the files work on any computer. Do not edit or re-split them: notebook Cell 1.0
checks their MD5 checksums.
