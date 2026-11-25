# Khmer Handwritten Consonant Recognition

A machine learning project that recognizes **handwritten Khmer consonants**. Given an image of one handwritten letter, the model predicts which of the **33 Khmer consonants** it is, with a confidence score.

```
[image of a handwritten ក]  →  model  →  "ក (ka)", 94% confident
```

> Intro to Machine Learning, 4th-year class project. Team of 3.

| Phase | Status |
|---|---|
| 1. Data preparation (`notebooks/01_prepare_dataset.ipynb`) | **Done** for the first 13 writers. More sheets being collected. |
| 2. Training (`notebooks/02_train.ipynb`) | Not started |
| 3. Evaluation and demo | Not started |

---

## Contents

1. [Quick start for teammates](#1-quick-start-for-teammates)
2. [Problem and scope](#2-problem-and-scope)
3. [Project layout](#3-project-layout)
4. [Collecting handwriting](#4-collecting-handwriting)
5. [Adding new scans](#5-adding-new-scans)
6. [Building the dataset](#6-building-the-dataset)
7. [Dataset format and split](#7-dataset-format-and-split)
8. [Models and training](#8-models-and-training)
9. [Evaluation](#9-evaluation)
10. [Demo app](#10-demo-app)
11. [Team, roles and timeline](#11-team-roles-and-timeline)
12. [Working together with Git](#12-working-together-with-git)
13. [Results](#13-results)
14. [Limitations](#14-limitations)
15. [Troubleshooting](#15-troubleshooting)
16. [References](#16-references)

---

## 1. Quick start for teammates

The cropped dataset is **not** in the repo (`outputs/` is in `.gitignore`). After cloning, you build it yourself. It takes about a minute.

```bash
git clone <repo-url>
cd TestCrop
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
jupyter notebook notebooks/01_prepare_dataset.ipynb    # or open it in VS Code
```

Run all cells. You now have `outputs/dataset/` and `outputs/dataset.zip`.

**Re-running is always safe.** The notebook deletes `outputs/dataset/` and rebuilds it from `data/`, so nothing is ever duplicated. Everyone gets the same images and the same split, because the split uses a fixed seed. Re-run it whenever you pull changes to `data/`.

## 2. Problem and scope

Handwriting recognition is well developed for Latin scripts, but there is very little public data for **modern handwritten Khmer**. Most existing Khmer datasets come from palm-leaf manuscripts. Khmer is also hard for machines: many consonants differ only by a small stroke or loop. Recognizing single characters is the first building block for Khmer handwriting OCR.

| | |
|---|---|
| Learning type | Supervised (every image has a known label) |
| Task | Multi-class image classification, 33 classes |
| Main method | Convolutional neural network (CNN) |
| Baselines | Logistic regression, random forest |

**In scope:** the 33 consonants, one per image, from scans or photos of paper, converted to 64×64 grayscale.
**Out of scope:** vowels, subscript consonants (coeng), diacritics, numerals, whole words, live pen-stroke input.

<details>
<summary>The 33 classes</summary>

| ID | Letter | Name | ID | Letter | Name | ID | Letter | Name |
|---|---|---|---|---|---|---|---|---|
| 0 | ក | ka | 11 | ឋ | ttha | 22 | ព | po |
| 1 | ខ | kha | 12 | ឌ | do | 23 | ភ | pho |
| 2 | គ | ko | 13 | ឍ | ttho | 24 | ម | mo |
| 3 | ឃ | kho | 14 | ណ | na | 25 | យ | yo |
| 4 | ង | ngo | 15 | ត | ta | 26 | រ | ro |
| 5 | ច | cha | 16 | ថ | tha | 27 | ល | lo |
| 6 | ឆ | chha | 17 | ទ | to | 28 | វ | vo |
| 7 | ជ | cho | 18 | ធ | tho | 29 | ស | sa |
| 8 | ឈ | chho | 19 | ន | no | 30 | ហ | ha |
| 9 | ញ | nho | 20 | ប | ba | 31 | ឡ | la |
| 10 | ដ | da | 21 | ផ | pha | 32 | អ | qa |

</details>

## 3. Project layout

```
TestCrop/
├── data/                          # in Git. Only Person A edits this folder.
│   ├── raw/                       # scans renamed scan_001 … (never edit or overwrite)
│   ├── manifest.csv               # one row per page: which writer, which sheet
│   └── rename_log.csv             # original name -> scan_###. NOT in Git (has names), Person A keeps it locally
├── notebooks/
│   ├── 01_prepare_dataset.ipynb   # raw scans -> cropped 64×64 dataset
│   └── 02_train.ipynb             # (planned) baselines, CNN, evaluation
├── outputs/                       # NOT in Git, built by notebook 01
│   ├── dataset/images/<id>_<name>/*.png
│   ├── dataset/labels.csv
│   ├── dataset/_debug/            # straightened pages, for checking crops only
│   └── dataset.zip                # for uploading to Google Colab
├── requirements.txt
└── README.md
```

## 4. Collecting handwriting

There is no public dataset of modern handwritten Khmer consonants, so we collect our own.

**The sheet:** one A4 page, a 5 × 7 grid (33 letter boxes plus 2 spare boxes), and **four black squares in the corners** that the code uses to find and straighten the page. Always print at **100% scale** ("Actual size", not "Fit to page").

**Rules for writers**
1. Black pen.
2. One letter per box, large and centred in the white area.
3. Don't touch the box border.
4. Write naturally. Don't copy the printed font.
5. Made a mistake? Leave the box empty and tell the collector.
6. Write your **writer code** (e.g. `W14`) in the header, **never your name**.

**Rules for scanning**
- Use a scanner app (Google Drive scan, Microsoft Lens) or a flatbed scanner.
- Page flat and upright, even light, no shadows, **all four corner squares visible**.
- iPhone `.HEIC` isn't supported. Set the camera to "Most Compatible" or export as JPG.

**Before you leave the writer, check the sheet:** all 33 boxes filled, corners visible. A missing box means that letter falls short.

**Get new people, not extra sheets from the same people.** The test set must be handwriting the model has never seen, so the number of *different writers* matters more than the number of images.

## 5. Adding new scans

Send every new scan to **Person A**. Only Person A changes files in `data/`, so writer IDs never clash.

Raw files are messy: one file can hold several people, one person can be spread over several files, and file names are random. Files are therefore **never named after people**. `data/manifest.csv` records who wrote which page.

1. Copy the file into `data/raw/` with the next free name (`scan_013.pdf`, `scan_014.jpg`, …). Use lowercase extensions.
2. Add `old name,new name` to `data/rename_log.csv` (Person A's local copy only. It's in `.gitignore`).
3. Add **one row per page** to `data/manifest.csv`:

   ```csv
   file,page,writer_id,sheet_no,note
   scan_013.jpg,1,w14,1,phone photo
   scan_014.pdf,1,w15,1,
   scan_014.pdf,2,w15,2,
   scan_014.pdf,3,w16,1,file holds 2 writers
   ```

   - A **new person** gets the next ID (`w14`, `w15`, …).
   - A person who **already has an ID** keeps it, with the next `sheet_no`.
   - Photos are one page each: `page` = 1.
   - **Never put names in `note`.**
4. Run notebook 01 again. It stops with a clear message if a file or page is missing or listed twice.

**Not sure whether two pages are by the same person?** Give them the **same** ID. Splitting one person into two IDs can put their handwriting in both train and test (data leakage, test score too high). Merging two people under one ID only keeps them in the same split, which is harmless.

## 6. Building the dataset

`notebooks/01_prepare_dataset.ipynb`. Run every cell from top to bottom. It never modifies anything in `data/`.

| Section | What it does |
|---|---|
| 1. Setup | Imports and paths |
| 2. Settings | `IMG_SIZE = 64`, split ratios, `SEED = 42`, `MIN_PER_CLASS = 30` |
| 3. Load and check the manifest | Stops if a file or page is missing or duplicated |
| 4. Helper functions | Loading, straightening, cropping |
| 5. Check the crop boxes on one page | Red boxes should sit inside the writing areas |
| 6. Crop every page | **Deletes and rebuilds `outputs/dataset/`**, lists empty boxes and skipped pages |
| 7. Split by writer | Assigns train / val / test |
| 8. Quality check | Bar chart of images per letter, grid of samples per letter |
| 9. Rebuild labels and zip | Drops rows for deleted images, writes `outputs/dataset.zip` |

**How the cropping works**
1. Each PDF page is rendered (photos are opened and phone rotation is fixed).
2. The four black corner squares are found.
3. A perspective transform straightens the page to a fixed size, whatever the tilt or scale of the scan. Straightened pages are saved to `_debug/`.
4. Each box's writing area is cut out at its known position.
5. The ink is picked out. Nearly empty boxes are skipped. The letter is cropped tight, centred with a margin, and resized to 64×64.

**Removing bad crops.** In section 8, note bad images (blank, crossed out, wrong letter, border lines), delete them from `outputs/dataset/images/`, then run section 9.

> ⚠️ Deletions are **lost** whenever section 6 runs again, and teammates never receive them because `outputs/` isn't in Git. Until this is automated, Person A keeps the list of deleted file names in the team chat and removes them again after every rebuild. A planned `data/exclude.csv` will let the notebook skip them automatically.

**Something looks wrong?** Open the page in `outputs/dataset/_debug/` (`<writer>_s<sheet>.png`). If the straightened page is skewed, rescan it.

## 7. Dataset format and split

```
outputs/dataset/
├── images/
│   ├── 00_ka/    w01_s1_00.png, w02_s1_00.png, …
│   ├── …
│   └── 32_qa/
└── labels.csv
```

- **Images:** 64×64 grayscale PNG, dark ink on white.
- **File name:** `<writer_id>_s<sheet_no>_<label_id>.png`, e.g. `w10_s3_07.png`.
- **`labels.csv` columns:** `filepath, label_id, letter, romanization, writer_id, sheet_no, source_file, source_page, split`

### How the split works

All images of one writer go into **one** split, so the test set is handwriting the model has never seen.

1. List all writer IDs, shuffle them with `SEED = 42`.
2. 15% of **writers** → validation, 15% → test, rest → train.

**Current split (13 writers, 627 images, 19 per letter):**

| Split | Writers | Images |
|---|---|---|
| Train | w01, w02, w04, w05, w06, w09, w11, w12, w13 | 396 (63%) |
| Val | w07, w08 | 66 (11%) |
| Test | w03, w10 | 165 (26%) |

The image shares aren't 70/15/15 because the ratio is counted by writers, and `w10` wrote 4 sheets.

> **Adding writers reshuffles the split.** Existing writers can move between train, val and test. Scores from before and after new data are not comparable. That's why the dataset is **frozen** on a fixed date (see the timeline) and not changed after that.

**Target:** 30 images per letter = 30 complete sheets. We have 19, so we need **~11 more writers, one sheet each**. More is better: 30+ writers gives 4–5 writers each in val and test.

## 8. Models and training

*Planned for `notebooks/02_train.ipynb`, to run on Google Colab (Runtime → Change runtime type → T4 GPU) or a laptop.*

| # | Model | Type | Purpose |
|---|---|---|---|
| 1 | Logistic regression | Classical, linear | Simplest baseline |
| 2 | Random forest | Classical, tree ensemble | Non-linear baseline |
| 3 | Simple CNN | Deep learning, from scratch | Our main model |
| 4 | ResNet-18 *(optional, if time allows)* | Pretrained on ImageNet, fine-tuned | Do pretrained features help with little data? |

**Why a CNN:** the baselines see an image as a flat list of pixels. A CNN learns local patterns (strokes, curves, loops) wherever they appear, which suits handwriting.

```
Input 1×64×64
→ Conv(32)  → BatchNorm → ReLU → MaxPool    → 32×32×32
→ Conv(64)  → BatchNorm → ReLU → MaxPool    → 64×16×16
→ Conv(128) → BatchNorm → ReLU → MaxPool    → 128×8×8
→ Flatten → Dense(128) → ReLU → Dropout(0.5) → Dense(33)
```

**Preprocessing (same for every model):** grayscale, scale to 0–1, invert so ink = 1 and paper = 0.

**Data augmentation (training only):** small rotations, shifts and zoom, e.g. `transforms.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1))`. With this little data, augmentation probably matters more than the choice of model.

| Setting | Simple CNN | ResNet-18 |
|---|---|---|
| Optimizer | Adam | Adam |
| Learning rate | 1e-3 | 1e-4 |
| Batch size | 32 | 32 |
| Epochs | 20–30 | 10–15 |
| Stopping | Keep the best epoch on validation | Same |

### Choosing settings: cross-validation by writer

Validation has only a few writers, so a single validation score is noisy. For tuning, use **5-fold cross-validation over the non-test writers**:

- `sklearn.model_selection.GroupKFold` (or `StratifiedGroupKFold`) with `groups = writer_id`, so **a writer never appears on both sides of a fold**.
- Baselines: `GridSearchCV(..., cv=GroupKFold(5))` and pass `groups` to `.fit()`. It tries every settings combination and reports the best.
- CNN: train once per fold for the 2–3 setups you want to compare. Report mean ± std.
- Final model: retrain on all non-test writers with the best settings.

> GridSearchCV picks the best **settings**, never the best **split**. Never try several splits and keep the one with the highest score. That only selects the easiest test writers.

**Rules:** tune on validation or CV only. Fixed random seeds. Plot training and validation curves for the report.

## 9. Evaluation

The **test set is used once, at the end**, after the model is final. Tuning on it makes the result meaningless.

**Metrics:** accuracy, top-3 accuracy, macro F1 (every letter counts equally), and a 33×33 confusion matrix for the best model.

**Error analysis**
- Find the 5 most confused letter pairs in the confusion matrix.
- Show 3–4 real misclassified images per pair.
- Explain why (the letters differ by one small stroke, an unusual writing style, a bad crop).

**Warning sign:** accuracy near 100% usually means data leakage, most often one writer in both train and test.

## 10. Demo app

*Optional, only if time allows.* A Gradio app where the user draws a letter or uploads a photo and sees the top 3 predictions.

1. Save the final model weights (`.pt`) and the 33 letters in label order.
2. Write `predict(image)` that preprocesses **exactly like the training data** (ink as foreground, crop to the ink, centre with a margin, 64×64, same scaling).
3. Gradio: sketchpad + upload + label output.
4. Record a backup screen video in case the internet fails during the presentation.

Mouse drawings look different from pen on paper. Use a thick brush and offer photo upload.

## 11. Team, roles and timeline

| Member | Role | Responsibilities |
|---|---|---|
| ___ | **A: Data** | Adding scans, `manifest.csv`, notebook 01, crop checks, removing bad crops, freezing the dataset, Data section of the report |
| ___ | **B: Model** | `02_train.ipynb`: baselines, CNN, augmentation, cross-validation, final model file, Methods section |
| ___ | **C: Evaluation and report** | Evaluation code, the single test run, confusion matrix, error analysis, Results and Limitations sections, slides, demo if time allows |

**Everyone collects handwriting.** Each of us fills in one sheet if we're not already in the data, then finds **3–4 new writers**.

**Handoffs:** A → frozen dataset · B → final model (`.pt`) and CV scores · C → test scores, figures, slides.

**B and C don't wait for the new data.** Build and debug on the current 627 images now, then re-run on the frozen dataset.

### Timeline (deadline Fri 4 Dec. Shift dates if it's earlier.)

| Day | Everyone | A: Data | B: Model | C: Evaluation and report |
|---|---|---|---|---|
| Wed 25 – Thu 26 Nov | Print sheets, fill one in, start asking people | Set up repo, clean names from manifest notes | Baselines on current data | Evaluation code, report and slides outline |
| Fri 27 – Sun 29 Nov | **Collect: all sheets in by Sunday night** | Add scans as they arrive, check crops | CNN + augmentation + CV | Run evaluation on validation |
| **Mon 30 Nov** | | **Freeze the dataset.** Nothing added after this. | | |
| Tue 1 – Wed 2 Dec | | Help B and C | Final training on the frozen dataset | Error analysis on validation, Methods text |
| **Thu 3 Dec** | | | **Model frozen** | **Run the test set once**, fill in Results |
| Fri 4 Dec | Rehearse and submit | | | |

If collection falls short by Sunday, **freeze anyway** and report it as a limitation. Don't move the freeze date.

## 12. Working together with Git

- **Keep the repo private.** Some scanned sheets may show a writer's name in the header. `rename_log.csv` (original file names, some are people's names) is kept out of Git on purpose.
- **What's in Git:** `data/`, notebooks, `requirements.txt`, README. **Not in Git:** `outputs/`. Rebuild it with notebook 01 after pulling.
- **Only Person A commits changes** to `data/` and notebook 01. Each notebook has one owner (02 → Person B), because notebooks store their output inside the file and two people committing the same one causes merge conflicts.
- **Pull before you start, push often.** Short commit messages saying what changed ("add scans for w14–w16").

## 13. Results

*To be filled in after training.*

| Model | CV accuracy (mean ± std) | Test accuracy | Top-3 | Macro F1 |
|---|---|---|---|---|
| Logistic regression | | | | |
| Random forest | | | | |
| Simple CNN | | | | |
| ResNet-18 (optional) | | | | |

**Dataset:** ___ writers, ___ sheets, ___ images (train ___ / val ___ / test ___)

**Most confused pairs:** ___

**Our goals** (targets, not known results): the CNN clearly beats both baselines, and the best model reaches at least 85% test accuracy.

## 14. Limitations

- **Few writers:** 13 so far, mostly students. Results may not generalize to other handwriting.
- **Small, uneven val and test sets:** each split covers only a few people, and one writer with 4 sheets dominates the current test set.
- **Writer identity is judged by hand** for files that mix people. A mistake could put one person in two splits and make the test score slightly optimistic.
- **Consonants only, single characters only.**
- **Fixed template:** cropping works only with our printed sheet at 100% scale.

**Future work:** more varied writers (ages, regions, pens), vowels and subscripts, whole words (CRNN or transformer OCR), pretraining on the SleukRith palm-leaf dataset.

## 15. Troubleshooting

| Problem | Fix |
|---|---|
| `zsh: command not found: python` | On Mac use `python3`. |
| `error: externally-managed-environment` | Use a virtual environment (see Quick start). |
| `ModuleNotFoundError` | Activate `.venv` first, or `pip install -r requirements.txt`. |
| "Run this from the project folder" | Open Jupyter from `TestCrop/` or `TestCrop/notebooks/`. |
| Notebook stops at the manifest check | A file in `manifest.csv` is missing from `data/raw/`, a page number is too high, or a page is listed twice. The message says which. |
| "corner squares not found, page skipped" | Rescan: flat, upright, even light, all four corners visible. |
| Many "empty box" messages | Pen too light, wrong print scale, or bad scan. Check section 5 and the `_debug/` image. |
| Border lines or cut-off letters | Print at 100% scale, ask writers to stay inside the white area. |
| My deleted bad images came back | Section 6 rebuilds everything. See the warning in [Building the dataset](#6-building-the-dataset). |

## 16. References

- He, K., Zhang, X., Ren, S., & Sun, J. (2016). *Deep Residual Learning for Image Recognition.* CVPR.
- Valy, D., Verleysen, M., Chhun, S., & Burie, J.-C. (2017). *A New Khmer Palm Leaf Manuscript Dataset for Document Analysis and Recognition: SleukRith Set.*
- Annanurov, B., & Noor, N. M. *A compact deep learning model for Khmer handwritten text recognition.* IAES International Journal of Artificial Intelligence.
- PyTorch: https://pytorch.org · scikit-learn: https://scikit-learn.org · Gradio: https://www.gradio.app · PyMuPDF: https://pymupdf.readthedocs.io
