# Task List: Khmer Consonant Classifier + Draw-to-Predict App

Full task breakdown for the 3-person team. For project background, data format and the dated timeline, see [README.md](README.md).

**Roles:** **A** = Data · **B** = Model · **C** = Evaluation and app

---

## Phase 0: Setup (all, 1–2 days)

- [ ] Agree on image size (64×64), color mode (grayscale), and `classes.json` with the 33 consonants (**All**)
- [ ] Agree on the split rule: by writer, stratified by class (**All**)
- [ ] Set up repo structure: `src/`, `models/`, `app/`, `reports/` (**C**)
- [ ] Branching rule, and `.gitignore` for large files (**C**)
- [ ] Commit the untracked scans and the manifest changes (**A**)

## Phase 1: Data (A)

- [ ] Finish collecting and cropping the scans
- [ ] Label the crops and QA `manifest.csv` (missing, duplicate, or wrong labels)
- [ ] EDA: class counts, sample grids, image-size stats, per-scan quality
- [ ] Collect more samples for weak classes
- [ ] Preprocess: grayscale, denoise, resize with padding, normalize
- [ ] Split by writer, stratified by class, and save `train.csv`, `val.csv`, `test.csv`
- [ ] Write `src/data.py` (load, preprocess, augment) and `classes.json`
- [ ] Write the data README section (class counts, image size, normalization mean/std, known issues)
- [ ] Collect 20–50 canvas drawings per class for the domain-gap test (Phase 5)
- [ ] Freeze the dataset on the agreed date

## Phase 2: Modeling (B)

- [ ] Baselines: logistic regression, random forest
- [ ] Training script with fixed seeds, logging, and checkpoints
- [ ] Simple CNN with augmentation
- [ ] Cross-validation grouped by writer (`GroupKFold`)
- [ ] Transfer learning (ResNet-18), if time allows
- [ ] Pick the final model, save the weights, export (ONNX or TorchScript)
- [ ] Write `predict(image) -> top-3 classes with confidences`
- [ ] Give C a dummy `predict()` early so the app can be built in parallel

## Phase 3: Evaluation (C)

- [ ] Evaluation script: accuracy, top-3 accuracy, macro F1, per-class precision/recall
- [ ] 33×33 confusion matrix and the most-confused letter pairs
- [ ] Error analysis: look at misclassified images, send suggestions to A and B
- [ ] Run the final test **once** on the held-out test set

## Phase 4: App (C, with B for the backend)

- [ ] **Drawing canvas:** mouse and touch input, adjustable brush, Clear and Undo
- [ ] **Canvas preprocessing:** must match training exactly (polarity, crop to ink, centre with margin, resize to 64×64, same scaling)
- [ ] **Backend API:** FastAPI `/predict` endpoint, returns top-3 predictions with confidences (**B**)
- [ ] **Results UI:** predicted consonant, confidence, top-3 list
- [ ] **Khmer font:** make sure the glyphs render correctly in the browser
- [ ] **Low-confidence handling:** "not sure, try again" below a threshold
- [ ] Photo upload as a fallback input
- [ ] *Quick alternative:* a Gradio app (sketchpad + upload + label output) if time is short

## Phase 5: Close the domain gap (A, B, C)

Canvas drawings look different from pen on paper (stroke width, smoothness, no paper noise).

- [ ] Test the model with drawings from team members and friends
- [ ] Add augmentation that mimics canvas strokes: thickness changes, blur (**B**)
- [ ] Use A's canvas drawings to test, or to fine-tune (**A**)
- [ ] Re-evaluate on canvas data and report the accuracy gap (**C**)

## Phase 6: Delivery (all)

- [ ] End-to-end integration test: draw → predict → display
- [ ] Deploy (Hugging Face Spaces, Render) or prepare to run locally for the demo
- [ ] Final report: problem, data, methods, results, error analysis, limitations
- [ ] Slides and a demo script
- [ ] Backup screen recording of the demo
- [ ] Clean up the repo and write run instructions

---

## Tasks per person

### Person A: Data
**Main job:** a clean, correctly split dataset that everyone trains and tests on.

- Finish collecting and cropping scans, QA the manifest
- EDA, preprocessing, and the writer-based split
- Write `src/data.py`, `classes.json`, and the data README section
- Collect canvas drawings for the domain-gap test
- **Later:** help with canvas preprocessing and the backend
- **Report:** Data and EDA sections

### Person B: Model
**Main job:** train the best model and provide a simple `predict()` function.

- Baselines, CNN, augmentation, cross-validation, optional ResNet-18
- Final model weights and export
- `predict(image) -> top-3`
- Canvas-style augmentation for the domain gap
- **Later:** build the FastAPI `/predict` endpoint
- **Report:** Methods and model-comparison sections

### Person C: Evaluation and app
**Main job:** measure the model and build the draw-to-predict app.

- Repo setup (structure, `requirements.txt`, `.gitignore`, branching rule)
- Evaluation script, confusion matrix, error analysis, the single test run
- Frontend: canvas, results display, Khmer font, low-confidence message
- Integration testing and deployment
- **Report and slides:** Results and App sections, demo script, backup recording

### Shared (all three)
- Phase 0 decisions
- Collect handwriting from new writers (see README section 4)
- Test the app with real drawings
- Final report, slides, and demo rehearsal

---

## Dependencies between people

| From | To | What |
|---|---|---|
| **A** | B, C | Splits and `src/data.py`. Hand over a first version early, even with partial data. |
| **B** | C | The `predict()` function. B gives C a dummy version first so the app can be built in parallel. |
| **A, B** | C (app) | Canvas preprocessing must match training preprocessing. Agree on this early. |

> **Note:** README section 11 lists the demo app as optional and assigns it to Person C "if time allows". This file treats the app as a core deliverable. If the team agrees, update README sections 10 and 11 to match.
