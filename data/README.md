# Datasets

| Folder | Used for | Source |
|---|---|---|
| `stone_seg/` | **Kidney stone localisation** (StoneUNet) | Your uploaded dataset (838 CT slices, 512×512, `image/*.tif` + stone masks `label/*.tif`). **Already included.** `splits.json` is already generated. |
| `kidney/` | **Kidney localisation (I2IRegNet) + stone detection (ClfNet / I2IReg-ClfNet)** | Paper dataset: Öksüz et al., *I2IReg-ClfNet*, BSPC 119 (2026) 109857 — Figshare **doi:10.6084/m9.figshare.31156783** (4,184 NCCT slices, 2,166 Stone+ / 2,018 Stone−, kidney masks drawn by three radiologists for every slice). |
| `ct_kidney_external/` | External test (paper Section 3.4.4.2) | CT KIDNEY DATASET (Islam et al., 2022) — Kaggle: `nazmul0087/ct-kidney-dataset-normal-cyst-tumor-and-stone`. Copy its `Normal/` and `Stone/` folders here. |

## Getting the paper dataset into `data/kidney/`

1. Download it from Figshare (doi:10.6084/m9.figshare.31156783). During peer review the paper says access was
   restricted — if the download is still locked, use the "Request access" button on Figshare or e-mail the
   corresponding author (C. Öksüz, cosku.oksuz@bakircay.edu.tr).
2. Extract it anywhere, e.g. `data/raw_kidney/`.
3. Run:

   ```bash
   python scripts/prepare_kidney_dataset.py --raw_dir data/raw_kidney
   python scripts/make_splits.py --dataset kidney
   ```

   The script finds image/mask pairs and Stone+/Stone− labels from the folder names. If the release uses a
   label spreadsheet instead, pass it with `--labels_csv labels.csv` (columns `filename,label[,patient_id]`).
   Check the printed counts (should be about 2,166 Stone+ / 2,018 Stone− and about 287 patients).

Resulting layout:

```
data/kidney/images/<name>.png    CT slice
data/kidney/masks/<name>.png     binary kidney mask
data/kidney/labels.csv           filename,label,patient_id   (1 = Stone+, 0 = Stone−)
data/kidney/splits.json          patient-level folds (5 CV + 1 hold-out)
```

**If the Figshare dataset is still locked:** any CT dataset with kidney masks and stone labels works through the same
script. For example, Zenodo record 6042410 ("An unenhanced abdominal CT dataset for kidney and kidney stone
segmentation") has both kidney and stone masks. Set label = 1 for slices that contain a stone.
