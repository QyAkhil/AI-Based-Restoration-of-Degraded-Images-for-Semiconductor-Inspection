# AI-Based Restoration of Degraded Images for Semiconductor Inspection

This project reproduces the notebook-based NAFNet restoration workflow as a clean submission-ready structure for the KLA challenge.

## Project structure

- `run.py` — required submission entry point
- `requirements.txt` — Python dependencies with version pins
- `README.md` — setup and execution instructions
- `models/` — trained checkpoint(s) used for inference
- `src/` — modular training and model code
- `data/` — training and test data folders

## Setup

```bash
pip install -r requirements.txt
```

## Training

```bash
python src/train.py --epochs 100 --batch_size 32 --lr 1e-4 --base_ch 32 --patch_size 128 --data_root data/train/train
```

## Inference for submission

The required entry script is:

```bash
python run.py <input-dir> <output-dir>
```

Use the actual folder that contains the input `.npy` files. In most cases, that means the folder directly containing the noisy images, for example:

```bash
python run.py data/test outputs
```

The script also handles the common nested layout where the files are under a `NoisyLR` subfolder, so it remains robust even if the folder structure is slightly different.

The script reads all `.npy` files in the input directory, creates the output directory if needed, and writes restored outputs with the same filenames.

## Notes

- The model expects grayscale noisy LR inputs and restores them to 2x resolution.
- Output arrays are saved as `.npy` files with shape `(H, W)` or `(H, W, 1)` and values in `[0, 1]`.
- The script automatically resolves the common `NoisyLR` layout when present, but the intended judge usage is to pass the folder containing the `.npy` files directly.
