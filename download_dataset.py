#!/usr/bin/env python3
"""
Download a Google Drive folder using gdown.
Usage:
    python download_dataset.py --gdrive_link <link> --output_dir <output_dir>
"""
import argparse
import sys
import subprocess

parser = argparse.ArgumentParser(description="Download a Google Drive folder using gdown.")
parser.add_argument('--gdrive_link', type=str, required=True, help='Google Drive folder link')
parser.add_argument('--output_dir', type=str, default='./pyronear_data_downloaded', help='Output directory')
args = parser.parse_args()

# Check if gdown is installed
try:
    import gdown
except ImportError:
    print("gdown is not installed. Installing now...")
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'gdown'])
    import gdown

print(f"Downloading Google Drive folder: {args.gdrive_link}")
gdown.download_folder(args.gdrive_link, output=args.output_dir, quiet=False, use_cookies=False)

print(f"\n✅ Download complete! Dataset is in: {args.output_dir}")
print("You can now use this folder as your dataset path for training.") 