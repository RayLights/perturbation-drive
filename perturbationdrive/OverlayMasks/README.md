# Overlay Masks

This directory contains image and video masks used for perturbations.

## Large Files Removed

To reduce repository size, the following large files have been removed from version control:

- **Video files**: `*.mp4` (lightning, rain, smoke, sun, snow, birds, test)
- **Large static masks**: `static_*.png` files (except min/max variants that remain)
- **MasksZip.zip**: Archive containing additional masks

## Download

If you need these files for your work, you can:

1. Download from the Google Drive (if available):
   https://drive.google.com/drive/folders/1_8v3NfX3j_holplmxNRuszirhGfUzVv4?usp=sharing

2. Or generate your own masks following the patterns in the existing files

Place downloaded mask files in this directory to use them with the perturbation functions.

## Note

Video files (*.mp4), MasksZip.zip, and static_*.png files are ignored by git as per .gitignore configuration.
