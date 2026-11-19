# IT8414 — Tomato Leaf Disease Classification

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Eman-271/IT8414-Tomato-Leaf-Disease-Classification/blob/main/notebook/IT8414_Tomato_Leaf_Classification_with_Member2.ipynb)

**Course:** IT8414 Deep Learning for Data Analytics · **Assessment 2 — Group Project** · **Group 01**

Three-class image classification of tomato leaves (**Healthy**, **Early Blight**,
**Late Blight**) using PlantVillage images. A custom CNN baseline (Member 1) is compared
with a transfer-learning model based on **MobileNetV2** (Member 2). Evaluation and
interpretability (Member 3, notebook sections 8.0 – 10.0) are in progress.

| Member | Role | Notebook sections |
|---|---|---|
| Member 1 | Data pipeline & baseline CNN | 1.0 – 5.9 |
| Member 2 | Improved model (MobileNetV2) & tuning | 6.0 – 7.7 |
| Member 3 | Evaluation & interpretability | 8.0 – 10.0 (to come) |

---

## 1. Repository structure

```
IT8414-Tomato-Leaf-Disease-Classification/
├── README.md
├── documentation/
│   └── DATA.md                   # dataset source, licence, structure, preparation (Deliverable 3)
├── requirements.txt
├── .gitignore
├── data/
│   ├── original/                 # put the Kaggle class folders here (not committed)
│   ├── selected/                 # balanced 300-image subset: 100 per class (seed 42)
│   │   ├── Tomato___Early_blight/
│   │   ├── Tomato___Late_blight/
│   │   └── Tomato___healthy/
│   └── split_manifests/          # the official 80/10/10 split used by every model
│       ├── train_split.csv       # 240 images (80 per class)
│       ├── validation_split.csv  # 30 images (10 per class)
│       ├── test_split.csv        # 30 images (10 per class)
│       └── data_summary.json     # counts and leakage-check results
├── scripts/
│   └── prepare_data.py           # rebuilds data/selected and the split manifests
├── notebook/
│   ├── IT8414_Tomato_Leaf_Classification.ipynb              # Member 1 (fixed): sections 1.0 – 5.9
│   ├── IT8414_Tomato_Leaf_Classification_with_Member2.ipynb # master notebook: Member 1 + Member 2
│   └── IT8414_Member2_Sections_6_7.ipynb                    # Member 2 sections only (runs standalone)
└── results/
    ├── baseline_runs.json                      # Member 1 baseline tuning runs (used by M1_REUSE_SAVED)
    ├── split_ratio_check_without_test.csv      # split-ratio check without the test images
    ├── improved_mobilenetv2_final.keras        # final improved model (Exp. I8)
    ├── improved_stage1_selected.weights.h5     # stage-1 selected weights (Exp. I3)
    ├── improved_runs.json                      # Member 2 tuning runs (used by M2_REUSE_SAVED)
    ├── improved_validation_results.json
    ├── improved_hyperparameter_results.csv
    ├── improved_layer_table.csv
    ├── improved_overfitting_ablation.csv
    └── figures/                                # report figures 3.4, 4.1, 5.2, 5.3, 5.4, D.1, D.2
```

## 2. Environment

Developed on **Google Colab** (T4 GPU runtime); Member 2's verification run was on CPU.
Random seed: **42** everywhere.

| Library | Version |
|---|---|
| Python | 3.13 |
| TensorFlow | 2.21.0 |
| Keras | 3.13.2 |
| NumPy | 2.4.6 |
| pandas | 2.2.3 |
| scikit-learn | 1.6.1 |
| Matplotlib, Pillow | latest |

```bash
pip install -r requirements.txt
```

## 3. Dataset

Source: *plantvillage Tomato leaf dataset* on Kaggle —
https://www.kaggle.com/datasets/charuchaudhry/plantvillage-tomato-leaf-dataset
(three classes: `Tomato___healthy` 1,273, `Tomato___Early_blight` 800, `Tomato___Late_blight` 1,527 images).

The repository already contains the balanced 300-image subset in `data/selected/` and the
official split in `data/split_manifests/`, so **you can run every notebook without downloading
anything**. Manifest columns: `filename, label, label_id, split, relative_path`.

To rebuild the subset from the full download (no TensorFlow needed):

```bash
# copy the three Kaggle class folders into data/original/ first
python scripts/prepare_data.py                 # same seed → same 300 images and split
python scripts/prepare_data.py --rebuild       # recreate data/selected from scratch
```

## 4. How to run the merged pipeline

The notebooks find the project folder in this order: the environment variable
`IT8414_PROJECT_DIR`, then (outside Colab) the folder above `notebook/`, then
`/content/IT8414_Tomato_Project` in Colab.

### Local / offline (Jupyter)

```bash
git clone https://github.com/Eman-271/IT8414-Tomato-Leaf-Disease-Classification.git
cd IT8414-Tomato-Leaf-Disease-Classification
pip install -r requirements.txt
cd notebook && jupyter notebook IT8414_Tomato_Leaf_Classification_with_Member2.ipynb
```

For the standalone Member 2 notebook, set the project folder first:
`export IT8414_PROJECT_DIR=$(pwd)/..` (Windows: `set IT8414_PROJECT_DIR=...`).

### Google Colab (one click)

1. Click the **Open in Colab** badge at the top of this page.
2. Runtime → Change runtime type → **T4 GPU**, then **Runtime → Run all**.

Nothing has to be uploaded. Cell 1.0 checks the official 80/10/10 split by the MD5 checksums of
its manifests; if `/content/IT8414_Tomato_Project` does not hold exactly that split, it downloads
this repository there (data, manifests, saved models). An old or wrong folder is moved aside,
never used, and the notebook never makes a new split silently.

### What runs where

| Sections | Content |
|---|---|
| 1.0 – 4.0 | Data loading, EDA, balanced subset, preprocessing, pipeline diagram |
| 5.1 – 5.4 | Split-ratio experiment (80/10/10 chosen) and final split |
| 5.5 – 5.9 | Six baseline CNN experiments, final baseline, one-time test evaluation |
| 6.0 – 7.7 | MobileNetV2 transfer learning: stage 1 (head) tuning I1–I7, stage 2 fine-tuning I8–I10, overfitting ablation |

**Reproducible numbers:** `M1_REUSE_SAVED = True` (Cell 5.2) loads the baseline runs from
`results/baseline_runs.json`; set `M2_REUSE_SAVED = True` (Cell 6.2) to load Member 2's runs from
`results/improved_runs.json` and the final model from `results/improved_mobilenetv2_final.keras`.
Set either flag to `False` to retrain. The final baseline model (`results/baseline_cnn_final.keras`)
is committed, so the baseline numbers are reproduced exactly.

## 5. Results summary (validation set)

| Model | Val accuracy | Val loss | Parameters |
|---|---|---|---|
| Baseline CNN (final) | 90.00% | 0.3779 | 101,699 |
| Improved MobileNetV2 (Exp. I8, fine-tuned from block 14, lr 1e-5) | 100.00% | 0.0176 | 2,261,827 (1,514,563 trainable) |

Baseline test result (Member 1): 80.00% accuracy, macro F1 79.38%, macro ROC-AUC 0.8950.
The test-set comparison of both models is Member 3's task (Cell 8.0).
All model-selection decisions used the validation set only.

## 6. Reproducibility measures

- One seed (42) for sampling, splitting, shuffling, augmentation and weight initialisation;
  reset before every model is built (`tf.keras.utils.set_random_seed`).
- The split is saved once as CSV manifests; every model loads the same manifests.
- Leakage checks: no exact or near-duplicate images across train / validation / test.
- The test set is used only for final, one-time evaluation.

## 7. Licence

The images come from the PlantVillage dataset, distributed under the
**Apache License 2.0** (http://www.apache.org/licenses/LICENSE-2.0). The 300 images in
`data/selected/` and the three class folders in `data/original/tomato_original.zip` are
unmodified copies, included under that licence with credit to Hughes & Salathé (2015).
