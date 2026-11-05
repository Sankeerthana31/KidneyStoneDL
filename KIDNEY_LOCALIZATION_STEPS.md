# Kidney localization - steps to do later

Run every command in the VS Code terminal with `(.venv)` showing, in `D:\KidneyStoneDL\KidneyStoneDL`.

## A. Get the kidney mask dataset (TotalSegmentator small subset, 3.2 GB, CC BY 4.0)
1. Download
   curl.exe -L -o data\totalseg_small.zip "https://zenodo.org/records/10047263/files/Totalsegmentator_dataset_small_v201.zip?download=1"
2. Extract
   Expand-Archive data\totalseg_small.zip -DestinationPath data\raw_totalseg
3. Check (should print about 102)
   (Get-ChildItem data\raw_totalseg -Recurse -Filter ct.nii.gz).Count
4. Install the 3D-scan reader
   pip install nibabel

## B. Prepare the training data
5. Convert to 2D kidney slices (15-20 min), then check data\kidney_masks\preview\
   python scripts/prepare_totalseg_kidney.py --raw_dir data/raw_totalseg
6. Make the splits
   python scripts/make_splits.py --dataset kidney --data data/kidney_masks

## C. Train kidney localization
7. Quick trial (2-3 min)
   python scripts/train_i2iregnet.py --fold 0 --data data/kidney_masks --epochs 2 --limit 64
8. Real training (about 2-2.5 hours on CPU)
   python scripts/train_i2iregnet.py --fold 0 --data data/kidney_masks --epochs 20

## D. Use the kidney region on Figshare
9. Predict kidney masks for the Figshare images, then check data\kidney\mask_preview\
   python scripts/predict_kidney_masks.py --reg outputs/i2iregnet/fold0/best.pt --data data/kidney
10. Stone detection inside the kidney region only (compare with 93.6% / 10 false alarms)
   python scripts/localize_figshare.py --ckpt outputs/stone_seg/fold0/best.pt --kidney_only

## E. App
11. Switch on the kidney model in app.py: set SHOW_KIDNEY_MODEL = True (line 22)

## Optional
12. Classifier on the kidney ROI (about 2.5 hours)
   python scripts/train_clfnet.py --fold 0 --input roi --epochs 20
