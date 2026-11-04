# Deep Learning Framework for Automated Kidney Stone Localization and Detection in Abdominal CT Images

A PyTorch project for VS Code with three stages:

| Stage | Task | Model | Dataset |
|---|---|---|---|
| 1 | **Kidney localisation** | **I2IRegNet**: DeepLabV3+-style image-to-image regression, ResNet18 backbone, grouped-conv ASPP | Paper dataset (CT + kidney masks) |
| 2 | **Kidney stone detection** (Stone+ / Stone−) | **ClfNet**: dual encoder (shallow CNN + frozen SqueezeNet) + CBAM, joined with stage 1 as the cascaded **I2IReg-ClfNet** | Paper dataset |
| 3 | **Kidney stone localisation** (where the stones are) | **StoneUNet**: ResNet34 U-Net at full 512×512 resolution, Focal-BCE + Dice loss | Uploaded dataset (`data/stone_seg`) |

Stages 1 and 2 follow *Öksüz et al., "I2IReg–ClfNet: a cascaded multi-task deep learning framework for ROI-aware
kidney stone detection in abdominal CT images", Biomedical Signal Processing and Control 119 (2026) 109857*.

At inference time, the stages run as one pipeline:

```
CT slice ─► I2IRegNet ─► kidney ROI (ROI_hat = M ⊙ x) ─► ClfNet ─► P(stone) + Grad-CAM
        └─► StoneUNet ─► stone probability map × dilated kidney ROI ─► stone boxes / centroid / area / side
```

Limiting stone localisation to the predicted kidney ROI removes false positives from calcifications in the
gallbladder, liver or ureter. This is the main idea of the paper.

---

## 1. Setup in VS Code

1. Unzip the project and open the `KidneyStoneDL` folder with **File → Open Folder**.
2. Install the recommended extensions when VS Code asks (Python, Pylance).
3. Open a terminal (**Ctrl+`**) and create a virtual environment:

   **Windows (PowerShell)**
   ```powershell
   python -m venv .venv
   .venv\Scripts\Activate.ps1
   ```
   **Linux / macOS**
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```
4. Install PyTorch for your hardware first, then the remaining packages:
   ```bash
   # NVIDIA GPU (recommended):
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
   # or CPU only:
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
   pip install -r requirements.txt
   ```
5. Select the interpreter: **Ctrl+Shift+P → Python: Select Interpreter → .venv**.
6. Run the self-test:
   ```bash
   python tests/test_setup.py
   ```

Every step below is also in the **Run and Debug** panel (Ctrl+Shift+D) as a numbered launch configuration.

> No GPU? You can run everything on Google Colab or Kaggle (as the paper did). Upload the folder and run the same commands.

---

## 2. Kidney stone localisation (uploaded dataset, ready to run)

The uploaded dataset is already in `data/stone_seg/` (838 slices) with patient-safe splits in `splits.json`
(consecutive slice IDs are kept together, which prevents leakage between train and test).

```bash
python scripts/train_stone_seg.py --fold 0                 # about 80 epochs; use --epochs 2 --limit 32 for a quick test
python scripts/evaluate.py --task stone --ckpt outputs/stone_seg/fold0/best.pt --save_examples 12
python scripts/predict.py --input data/stone_seg/image/1000.tif --stone outputs/stone_seg/fold0/best.pt
```

Metrics: pixel **Dice / IoU** and stone-level **precision / recall / F1** (a ground-truth stone counts as found if a
predicted component overlaps it). In the example images, green = ground truth and red = prediction.

## 3. Kidney localisation + stone detection (paper dataset)

Get the dataset first (see `data/README.md`). Then:

```bash
python scripts/prepare_kidney_dataset.py --raw_dir data/raw_kidney
python scripts/make_splits.py --dataset kidney

# Stage-wise training (paper Section 3.4.1 and 3.4.2)
python scripts/train_i2iregnet.py --fold 0                 # kidney ROI regression: MSE/RMSE/MAE/IoU/Dice
python scripts/train_clfnet.py --fold 0                    # ShN + SqueezeNet + CBAM on ROIs

# Cascaded I2IReg-ClfNet (paper Section 3.4.3): L = 0.8·L_reg + 0.2·L_clf
python scripts/train_cascade.py --fold 0 \
    --init_reg outputs/i2iregnet/fold0/best.pt \
    --init_clf outputs/clfnet/ShN+squeezenet+CBAM_roi/fold0/best.pt

python scripts/evaluate.py --task cascade --ckpt outputs/cascade/fold0/best.pt   # confusion matrix + ROC
python scripts/run_cv.py --script train_cascade.py                               # 5-fold CV, mean ± std
```

Ablations from the paper:

```bash
python scripts/train_clfnet.py --deep_encoder resnet18      # ShN + RN18 + CBAM
python scripts/train_clfnet.py --no_shallow                 # single encoder
python scripts/train_clfnet.py --no_cbam                    # without attention
python scripts/train_clfnet.py --input image                # no ROI (full slice), like the non-ROI baselines
```

External test on the CT KIDNEY DATASET (paper Section 3.4.4.2):

```bash
python scripts/evaluate_external.py --ckpt outputs/cascade/fold0/best.pt              # without fine-tuning
python scripts/evaluate_external.py --ckpt outputs/cascade/fold0/best.pt --finetune   # 50 % fine-tune / 50 % test
```

## 4. Full pipeline and demo app

```bash
python scripts/predict.py --input path/to/slice_or_folder \
    --cascade outputs/cascade/fold0/best.pt --stone outputs/stone_seg/fold0/best.pt
streamlit run app.py
```

Each result image shows four panels: **CT | kidney ROI | Grad-CAM with P(stone) | detected stones**. The file
`results.json` lists, for each stone: bounding box, centroid, area in pixels, side (patient's right/left kidney),
and area in mm² when the input is DICOM.

---

## 5. Project structure

```
KidneyStoneDL/
├── app.py                       Streamlit demo
├── configs/config.yaml          all hyper-parameters (paper values: Adam, lr 1e-4, 50 epochs, 224×224)
├── data/
│   ├── stone_seg/               uploaded dataset (image/, label/, splits.json)
│   ├── kidney/                  paper dataset after prepare_kidney_dataset.py
│   └── ct_kidney_external/      CT KIDNEY DATASET (Normal/, Stone/)
├── scripts/
│   ├── prepare_kidney_dataset.py
│   ├── make_splits.py           patient-level 5-fold CV + hold-out
│   ├── train_i2iregnet.py       stage 1
│   ├── train_clfnet.py          stage 2
│   ├── train_cascade.py         joint I2IReg-ClfNet
│   ├── train_stone_seg.py       stone localisation
│   ├── evaluate.py / evaluate_external.py / run_cv.py
│   └── predict.py               full inference pipeline
├── src/
│   ├── models/{i2iregnet, clfnet, cascade, stone_unet, common}.py
│   ├── datasets.py, splits.py, metrics.py, gradcam.py, pipeline.py, io_utils.py, utils.py, engine.py
└── tests/test_setup.py
```

## 6. Implementation notes (paper → code)

* **I2IRegNet** (`src/models/i2iregnet.py`): ResNet18 FEB at output stride 16 (14×14), ASPP with grouped 3×3
  convs at rates 1/6/12/18 plus pointwise 256 → 1024 channels, up-sampling unit 1 (3×3 conv 256, transposed conv
  8×8 stride 4 → 56×56, concatenated with 48-channel low-level features → 304), up-sampling unit 2 (grouped +
  pointwise convs, 1×1 mapping to 3 channels, transposed conv → 224×224). The output map is multiplied with the
  input along the "reference line", so ROI_hat = M ⊙ x. About 11 M parameters.
* **ClfNet** (`src/models/clfnet.py`): shallow encoder from the paper's Bayesian search (kernels 128-32-96-64-256-224;
  blocks that would go below 7×7 are pruned). The frozen ImageNet deep encoder is resampled to 7×7, concatenated,
  then BN → 1×1 conv to round((Cs+Cd)/2) → CBAM → ReLU → Dropout → GAP → FC.
* **Loss**: L_reg = half-MSE(ROI, ROI_hat), L_clf = cross-entropy, L = 0.8·L_reg + 0.2·L_clf.
* **Metrics**: Accuracy, Recall, Specificity, Precision, F1, ROC-AUC and Cohen's κ for detection;
  MSE, RMSE, MAE, IoU and Dice for localisation.
* **Not specified in the paper (choices made here)**: the ROI binarisation threshold (0.05), CBAM reduction 16,
  dropout 0.5, data augmentation, and the plateau LR schedule.

Research prototype. Not a medical device.
