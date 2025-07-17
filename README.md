# Wildfire Detection with YOLO for Raspberry Pi

## Dataset

**Note:** The training dataset is NOT included in this repository and should NOT be uploaded to GitHub.

To use the code, you have two options:

### 1. Download the dataset automatically

Run the provided script to download the dataset from Google Drive:

```bash
python download_dataset.py --gdrive_link "https://drive.google.com/drive/folders/1-6NR3M9qBi7FTpEt0rhRQeEWD_33mBGH?usp=sharing" --output_dir "./pyronear_data_downloaded"
```

**Note:** If you encounter disk quota issues, extract to a location with more space:
```bash
unzip pyronear2025.zip -d /vol/bitbucket/mst124/pyronear_data_downloaded
```

### 2. Download manually

Go to [Google Drive Wildfire Dataset](https://drive.google.com/drive/folders/1-6NR3M9qBi7FTpEt0rhRQeEWD_33mBGH?usp=sharing), download the folder, and extract it to a location with sufficient disk space.

Then, use the local path as the dataset path for training.

---

## Training

After downloading, train with:

```bash
cd raspberry_pi_yolo
python train.py --pyronear_path "/vol/bitbucket/mst124/pyronear_data_downloaded/pyronear2025" --epochs 100
```

**Alternative paths:**
- If extracted locally: `--pyronear_path "./pyronear_data_downloaded/pyronear2025"`
- If extracted elsewhere: `--pyronear_path "/path/to/your/extracted/pyronear2025"`

---

## Why is the dataset not included?
- Datasets are large and not suitable for version control.
- This keeps the repository lightweight and avoids copyright/licensing issues.
- You can always fetch the latest dataset as needed. 