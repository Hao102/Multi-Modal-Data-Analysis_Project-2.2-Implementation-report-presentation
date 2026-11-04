Author: Tran Huu Hao Nguyen
# Multimodal Movie Genre Classification

This project predicts 23 movie genres using plot summaries and posters
from Simplified MM-IMDb. It compares text-only, image-only, early fusion
and three late fusion variants: equal, global and per-genre weighting.

## Main notebook

`Project 2.2 _ Code.ipynb`

The notebook covers data quality checks, preprocessing, feature
extraction, classifier training, validation and final test evaluation.

## Requirements

Required Python packages:
- numpy
- pandas
- matplotlib
- scikit-learn
- torch
- transformers
- pillow
- psutil
- jupyterlab

Install the packages in your Python environment:

```bash
python -m pip install numpy pandas matplotlib scikit-learn torch transformers pillow psutil jupyterlab
```

The reported experiments used Windows, CPU execution, eight PyTorch
threads, PyTorch 2.13.0+cpu and Transformers 4.57.6.
The installation command above does not pin package versions;
use the recorded versions for closer reproduction.

## Dataset

Download Simplified MM-IMDb from:
https://www.kaggle.com/datasets/javierurea/simplified-mm-imdb

Place `archive.zip` in the notebook's working directory.
The archive must contain `data.npy` and `images.npz`.

Alternatively, update `ZIP_PATH` in the first code cell to the full
path of your downloaded archive.

## How to run

1. Download this repository and install the required packages.
2. Place the dataset archive in the working directory.
3. Start JupyterLab using `jupyter lab`.
4. Open `Project 2.2 _ Code.ipynb`.
5. Run all cells in order within the same kernel session.

An internet connection is required for the initial download of the
DistilBERT and CLIP Vision models. Both encoders remain frozen.
Feature extraction on a CPU may take substantial time; completed
features are cached for reuse.

## Outputs and evaluation

Generated features and experiment results are saved under
`mmimdb_features/v2_317223387d28dbac/`.

The grouped split contains 15,581 training, 2,595 validation and
7,783 test movies. Checkpoints, thresholds and fusion weights are
selected using validation data before final test evaluation.

Reported early fusion test results:
- Macro-F1: 0.5552
- Micro-F1: 0.5995

Results and runtime may vary with hardware and software versions.
