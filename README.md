# Wildfire Detection with YOLO for Raspberry Pi

## Dataset

**Note:** The training dataset is NOT included in this repository and should NOT be uploaded to GitHub.

To use the code, you have two options:

### 1. Download the dataset automatically

Run the provided script to download the dataset from Google Drive:

```bash
python download_dataset.py --gdrive_link "https://drive.google.com/drive/folders/1-6NR3M9qBi7FTpEt0rhRQeEWD_33mBGH?usp=sharing" --output_dir "./pyronear_data_downloaded"
```

### 2. Download manually

Go to [Google Drive Wildfire Dataset](https://drive.google.com/drive/folders/1-6NR3M9qBi7FTpEt0rhRQeEWD_33mBGH?usp=sharing), download the folder, and extract it outside the repository.

Then, use the local path as the dataset path for training.

---

## Training

After downloading, train with:

```bash
python train.py --pyronear_path "./pyronear_data_downloaded/Pyro25" --epochs 100
```

(Adjust the path to the correct subfolder if needed.)

---

## Why is the dataset not included?
- Datasets are large and not suitable for version control.
- This keeps the repository lightweight and avoids copyright/licensing issues.
- You can always fetch the latest dataset as needed. 