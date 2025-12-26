# Repository Bloat Removal Summary

## Overview
This document summarizes the changes made to reduce the repository size and make it faster to clone.

## Files Removed

### Simulator Binaries (~300MB)
- `examples/udacity/sim/udacity_linux/` - Unity simulator binaries for Udacity
- `examples/self_driving_sandbox_donkey/sim/sdsim_linux/` - Unity simulator binaries for Donkey

**Rationale**: These large binary files should be downloaded separately by users who need them.

### Model Checkpoints (~116MB)
- `checkpoints/original_extended3_extended5.h5` (58MB)
- `examples/models/checkpoints/dave_90k_v1.h5` (58MB)

**Rationale**: Model checkpoint files are large and should be downloaded on-demand by users who need them for training/inference.

### Video Mask Files (~80MB)
- `perturbationdrive/OverlayMasks/test.mp4` (22MB)
- `perturbationdrive/OverlayMasks/lightning.mp4` (19MB)
- `perturbationdrive/OverlayMasks/rain.mp4` (15MB)
- `perturbationdrive/OverlayMasks/smoke.mp4` (8.9MB)
- `perturbationdrive/OverlayMasks/sun.mp4` (6.2MB)
- `perturbationdrive/OverlayMasks/snow.mp4` (2.7MB)
- `perturbationdrive/OverlayMasks/birds.mp4` (1.5MB)

**Rationale**: Large video files for dynamic perturbations can be downloaded separately.

### Static Mask Images (~15MB)
- `perturbationdrive/OverlayMasks/static_snow.png` (4.0MB)
- `perturbationdrive/OverlayMasks/static_rain.png` (2.9MB)
- `perturbationdrive/OverlayMasks/static_snow_max.png` (1.8MB)
- `perturbationdrive/OverlayMasks/static_sun.png` (1.3MB)
- `perturbationdrive/OverlayMasks/static_rain_max.png` (1.1MB)

**Rationale**: Large static mask images can be downloaded separately or regenerated.

### MasksZip Archive (177MB)
- `perturbationdrive/OverlayMasks/MasksZip.zip` (177MB)

**Rationale**: This large archive should be downloaded separately by users who need all masks.

### Jupyter Notebooks with Outputs (~100MB)
- `examples/models/benchmark_models.ipynb` (91MB)
- `examples/open_sbt/donkey_dave2_experiment.ipynb` (5.6MB)
- `perturbationdrive/RoadGenerator/Roads/road_experiments.ipynb` (2.3MB)
- `examples/models/benchmark_attention.ipynb` (1.9MB)

**Rationale**: Jupyter notebooks with embedded outputs bloat the repository. Users can re-run notebooks to generate outputs.

### Analysis Data (~5MB)
- `ICST2025_analysis/` directory (entire directory removed)

**Rationale**: Research analysis notebooks and data are not essential for the core library functionality.

### Documentation Images (~6MB)
- `perturbed_outputs.png` (5.6MB)
- `0001_0.png` (60KB)

**Rationale**: Large example output images can be regenerated or replaced with smaller versions.

## Total Size Reduction
**Removed from version control**: ~800MB worth of binary files
**Files deleted**: 380 files
**Lines removed**: 467,271 deletions

## Changes Made

### Updated .gitignore
Added patterns to prevent large files from being re-committed:
- Simulator binaries
- Model checkpoints (*.h5)
- Video mask files (*.mp4)
- MasksZip.zip
- Static mask images (static_*.png)
- Jupyter notebooks (*.ipynb)
- Analysis directories
- Large documentation images

### Added README Files
Created documentation in key directories explaining how to download required files:
- `checkpoints/README.md`
- `examples/models/checkpoints/README.md`
- `perturbationdrive/OverlayMasks/README.md`

### Updated Main README
Added a new section "Downloading Required Assets" explaining:
- Where to download simulator binaries
- Where to download model checkpoints
- Where to download overlay masks
- How to place downloaded files in the correct directories

## Download Location
All removed files are available for download from:
https://drive.google.com/drive/folders/1_8v3NfX3j_holplmxNRuszirhGfUzVv4?usp=sharing

## Impact on Users

### Positive
- **Faster cloning**: Repository is now significantly smaller
- **Lower bandwidth usage**: Less data to download
- **Faster git operations**: Smaller repository means faster git operations

### Considerations
- Users who need simulators, model checkpoints, or large masks must download them separately
- Jupyter notebooks will not have pre-computed outputs (users must re-run them)
- Research analysis data is no longer in the repository

## Core Functionality
The core library functionality remains intact:
- All perturbation functions are still available
- All simulator integration code is present
- All example code and documentation is present
- Only large binary assets have been removed

## Note on Git History
The current .git directory still contains the history with these large files. To fully reduce the repository size, a git history rewrite would be needed (e.g., using `git filter-repo` or `BFG Repo-Cleaner`). This is a more invasive operation that was not performed as part of this change to preserve the git history.
