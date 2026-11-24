# DATA.md — Dataset Documentation

**Project:** IT8414 Deep Learning for Data Analytics — Tomato Leaf Disease Classification
**Group:** 01 · **Prepared by:** Fatema (Member 1 – Data Pipeline & Baseline Model)
**Last updated:** 24/11/2026

This file is the dataset deliverable required by the brief (Deliverable 3): the
download link, the licence, the folder structure and the preparation steps. The
repository also holds the data itself, so the notebook runs in Colab with no upload:
`data/original/tomato_original.zip` (the three original class folders), `data/selected/`
(the 300-image subset) and `data/split_manifests/` (the official 80/10/10 split).
Anyone can also rebuild the exact same dataset from Kaggle with the steps in Section 6.

---

## 1. Source

| Item | Detail |
|---|---|
| Kaggle dataset | **plantvillage Tomato leaf dataset** |
| Kaggle owner | charu (`charuchaudhry`) |
| Link | https://www.kaggle.com/datasets/charuchaudhry/plantvillage-tomato-leaf-dataset |
| Folder used | `plantvillage/` |
| Last updated on Kaggle | 21 March 2019 |
| Download size | 230.6 MB |
| Original collection | PlantVillage (Hughes & Salathé, 2015), https://doi.org/10.48550/arXiv.1511.08060 |
| How images were taken | One detached leaf per photo, on a plain grey background, in controlled lighting |
| Image format | RGB JPEG, 256 × 256 pixels |

## 2. Licence and ethical use

- **Licence: Apache License 2.0** (http://www.apache.org/licenses/LICENSE-2.0),
  the licence of the PlantVillage dataset, of which these tomato classes are part.
- The tutor (Dr. Shomona Jacob) confirmed that this dataset and licence may be
  used for the project.
- The images are kept, **unmodified**, in the team's public GitHub repository so that the
  notebook can run anywhere without an upload. Apache 2.0 allows redistribution with the
  licence and credit given; both are stated here and in the README.
- The data is used for **academic work only**, and the original PlantVillage
  authors (Hughes & Salathé, 2015) and the Kaggle uploader are credited.
- The images show plant leaves only. They contain no people and no personal data,
  so there are no privacy concerns.

## 3. Classes used and why

| Label ID | Folder name | Display name | Cause |
|---|---|---|---|
| 0 | `Tomato___healthy` | Healthy | – |
| 1 | `Tomato___Early_blight` | Early Blight | Fungus *Alternaria solani* |
| 2 | `Tomato___Late_blight` | Late Blight | Oomycete *Phytophthora infestans* |

**Why these three:**
1. **Early Blight and Late Blight look alike.** Both produce brown, dead
   (necrotic) patches on the leaf. Early Blight usually shows darker spots with
   ring patterns. Late Blight usually shows larger, paler, water-soaked patches,
   often starting at the leaf edge. Telling them apart is a **fine-grained** problem.
2. **Healthy is the reference class.** A screening tool must first decide
   whether a leaf is diseased at all.
3. Together they make a **three-class problem with one easy boundary**
   (healthy vs diseased) **and one hard boundary** (Early vs Late Blight). This
   is a meaningful test of what a CNN learns, without needing all ten tomato classes.

**Evidence (notebook Cell 3.0, Figure 3.3, `results/class_choice_evidence.json`).**
On 150 random images per class, the brown/yellow (lesion-coloured) share of the
leaf is 0.8% for Healthy, 14.7% for Early Blight and 18.7% for Late Blight. The
colour profiles of the two blights are the most similar pair (distance 0.261, vs
0.447 for Healthy–Early Blight and 0.294 for Healthy–Late Blight). A colour-only
classifier recognised 140/150 Healthy leaves, but called 50 Early Blight leaves
Late Blight and 62 Late Blight leaves Healthy. Colour alone cannot separate the two
blights, so the task is genuinely fine-grained.

## 4. Data characteristics (from notebook cells 2.0 and 3.0)

| Class | Images in source | Share | Selected |
|---|---|---|---|
| Healthy | 1,273 | 35.4% | 100 |
| Early Blight | 800 | 22.2% | 100 |
| Late Blight | 1,527 | 42.4% | 100 |
| **Total** | **3,600** | 100% | **300** |

**The 300-image balanced subset was approved by the tutor (Dr. Shomona Jacob) at the proposal stage.**

- **Imbalance:** Late Blight has 1.91 times as many images as Early Blight.
- **Note on counts:** these counts are about 80% of the class sizes in the
  original PlantVillage release (1,591 / 1,000 / 1,909). This Kaggle copy
  therefore appears to hold a subset of the original images.
- **Quality checks:** all 300 selected files open without error, all are
  256 × 256 RGB, and no labels are missing (labels come from folder names).

### Known limitations and bias
- **Laboratory conditions:** uniform grey background and controlled lighting. A
  model can learn background or lighting cues instead of disease symptoms.
  Mohanty et al. (2016) reported accuracy dropping from over 99% to about 31%
  when PlantVillage-trained models were tested on photos taken under other conditions.
- **One leaf per image, detached:** real field photos show several leaves, soil,
  shadows and partial views.
- **Labels are trusted as given:** they were not re-checked by a plant pathologist.
- **Small subset (300 images):** limits how much natural variation the models see.

## 5. Project folder structure

```
IT8414-Tomato-Leaf-Disease-Classification/     # in Colab: /content/IT8414_Tomato_Project/
├── data/
│   ├── original/                 # untouched Kaggle class folders (tomato_original.zip)
│   │   ├── Tomato___healthy/
│   │   ├── Tomato___Early_blight/
│   │   └── Tomato___Late_blight/
│   ├── selected/                 # balanced 100-per-class subset (generated)
│   │   ├── Tomato___healthy/
│   │   ├── Tomato___Early_blight/
│   │   └── Tomato___Late_blight/
│   └── split_manifests/          # the official split - every model uses these
│       ├── train_split.csv
│       ├── validation_split.csv
│       ├── test_split.csv
│       └── data_summary.json     # written by notebook Cell 5.4 or scripts/prepare_data.py
├── documentation/DATA.md         # this file
├── notebook/IT8414_Tomato_Leaf_Classification_Final.ipynb
├── scripts/prepare_data.py       # rebuilds data/selected and the manifests
└── results/                      # saved models, figures and tables
```

### Split manifest format

| Column | Example | Meaning |
|---|---|---|
| `filename` | `0a8a68ee-....JPG` | Image file name |
| `label` | `Tomato___Early_blight` | Class folder name |
| `label_id` | `1` | Integer label used for training (0, 1, 2) |
| `split` | `train` | `train`, `validation` or `test` |
| `relative_path` | `data/selected/Tomato___Early_blight/0a8a....JPG` | Path from the project root |

The manifests store **relative** paths, so they work on any computer or Google
Drive account. The notebook rebuilds the full path from `label` + `filename`.
Older manifests that only contain `image_path` and `label` are also accepted.

## 6. How to reproduce the dataset

1. Download the Kaggle dataset (link above) and unzip it.
2. Copy the three class folders from `plantvillage/` into `data/original/`.
3. Either run notebook sections **1.0–5.4** (the split ratio is chosen in 5.3), or run the script from the project root:

```bash
pip install numpy pandas pillow scikit-learn
python scripts/prepare_data.py            # keeps an existing balanced subset
python scripts/prepare_data.py --rebuild  # recreates it from scratch
python scripts/prepare_data.py --ratio 70/15/15   # another ratio (default 80/10/10)
```

Both use the same steps and the same seed (42), so they produce the **same 300
images and the same split**. This was verified on 11/11/2026: the script
reproduces the notebook's 80/10/10 manifests file for file.

## 7. Preparation steps and reasons

| # | Step | Setting | Reason |
|---|---|---|---|
| 1 | Class selection | Healthy, Early Blight, Late Blight | See Section 3 |
| 2 | Balanced sampling | 100 per class, `random.Random(42)`, files sorted first | Removes the 1.91 imbalance from the model comparison; keeps training short enough for repeated tuning runs on Colab. **The 300-image balanced subset was approved by the tutor at the proposal stage.** |
| 3 | Quality check | Open + verify every file; record sizes and colour modes | Catch corrupt or unusual files before training |
| 4a | Split-ratio experiment | 70/15/15 vs 80/10/10 vs 60/20/20; three random stratified splits each (seeds 42, 7, 2024); reference baseline trained and scored on **validation only** | The ratio is chosen by evidence, not assumed (see below) |
| 4b | Stratified split | **80 / 10 / 10 (240 / 30 / 30)**, `make_split()`: `train_test_split(..., stratify=label, random_state=42)` twice | Each split keeps exactly one third per class |
| 5 | Exact leakage check | Compare file paths across all split pairs | No image may appear in two splits |
| 6 | Near-duplicate check | 64-bit difference hash, flag pairs differing in ≤ 5 bits | Catches the same leaf saved under different names |
| 7 | Save manifests | `data/split_manifests/*.csv` | Every model and member uses identical images |
| 8 | Decode | RGB, 3 channels | Consistent tensor shape; colour carries lesion information |
| 9 | Resize | 256 × 256 → 224 × 224 | Standard input size of ImageNet pre-trained models, so the baseline and improved model share one pipeline |
| 10 | Normalise | Divide by 255 → range 0–1 | Stable gradients. Stateless, so no statistics come from validation or test data |
| 11 | Cache, shuffle, batch, prefetch | Cache in memory; shuffle training only (seed 42); batch 16; prefetch | Efficient `tf.data` input pipeline |
| 12 | Augmentation (training only) | Horizontal flip, rotation 0.1 (≈ ±36°), zoom ±10% | Leaf orientation carries no class information; more variety for 240 training images. Keras applies these layers only during training |

### Split-ratio experiment (notebook Cell 5.3, `results/split_ratio_comparison.csv`)

| Ratio | Train / validation images | Mean validation loss ± SD | Mean validation accuracy |
|---|---|---|---|
| 70/15/15 | 210 / 45 | 0.4581 ± 0.0044 | 79.26% |
| **80/10/10** | **240 / 30** | **0.4097 ± 0.0353** | **80.00%** |
| 60/20/20 | 180 / 60 | 0.4588 ± 0.0281 | 81.11% |

Rule fixed before the run: lowest mean validation loss wins; a ratio within one
standard error (SD/√3) of it counts as a tie, and then the ratio with larger
validation and test sets is preferred. Both alternatives were more than one
standard error worse, so **80/10/10 was chosen**. The candidate splits were drawn
independently of the final split, so their validation parts held up to 15 of the 30
final test images. Repeating the comparison on the other 270 images (Cell 5.3,
`results/split_ratio_check_without_test.csv`) chose 80/10/10 again (mean validation
loss 0.3591 vs 0.4402 and 0.4312), so the overlap did not influence the choice. The
trade-off is a 30-image test set (one image = 3.3 percentage points).

### Split sizes

| Split | Healthy | Early Blight | Late Blight | Total |
|---|---|---|---|---|
| Train | 80 | 80 | 80 | 240 |
| Validation | 10 | 10 | 10 | 30 |
| Test | 10 | 10 | 10 | 30 |

### Leakage results

| Check | Train / Val | Train / Test | Val / Test |
|---|---|---|---|
| Exact file overlap | 0 | 0 | 0 |
| Near-duplicate pairs (≤ 5 / 64 bits) | 0 | 0 | 0 |

Both checks found **no leakage** (notebook Cell 5.4, final run of 11 November 2026). All counts are also
saved in `data/split_manifests/data_summary.json`.

*If a future re-selection ever finds a near-duplicate pair across two splits, move both images
into the same split (or remove one), save the manifests again and note it here.*

## 8. Rules for all team members

- **Do not re-split the data.** Load the manifests (notebook Cell 4.0 does this automatically when `data/split_manifests/` exists).
- **Tune only on the validation set.** Use the test set once per final model.
- **Use `make_dataset()`** from notebook section 4.0 so every model gets the same preprocessing.
- Class order is fixed: **0 = Healthy, 1 = Early Blight, 2 = Late Blight**.

## 9. References

Apache Software Foundation. (2004). *Apache License, Version 2.0*. http://www.apache.org/licenses/LICENSE-2.0

charu. (2019). *plantvillage Tomato leaf dataset* [Data set]. Kaggle. https://www.kaggle.com/datasets/charuchaudhry/plantvillage-tomato-leaf-dataset

Hughes, D. P., & Salathé, M. (2015). *An open access repository of images on plant health to enable the development of mobile disease diagnostics* (arXiv:1511.08060). arXiv. https://doi.org/10.48550/arXiv.1511.08060

Mohanty, S. P., Hughes, D. P., & Salathé, M. (2016). Using deep learning for image-based plant disease detection. *Frontiers in Plant Science, 7*, Article 1419. https://doi.org/10.3389/fpls.2016.01419
