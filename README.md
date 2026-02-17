# Leaf Classification - CNN Training

This project trains an image classification model using TensorFlow/Keras on a leaf dataset.

## Project Structure

```text
NN/
|-- dataset/
|   |-- train/
|   |   |-- class_1/
|   |   |-- class_2/
|   |-- test/
|       |-- class_1/
|       |-- class_2/
|-- src/
|   |-- train.py
|-- requirements.txt
|-- .gitignore
```

## Prerequisites

- Python 3.10+ (recommended)
- `pip`
- Dataset folders available locally at:
  - `dataset/train`
  - `dataset/test`

## Setup

1. Create and activate a virtual environment.

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

2. Install dependencies.

```powershell
pip install -r requirements.txt
```

## Usage

Run training:

```powershell
python src/train.py
```

What `src/train.py` currently does:
- Loads images from `dataset/train` and `dataset/test`
- Resizes images to `224x224`
- Trains a CNN for `5` epochs
- Uses categorical classification based on folder names

## Notes

- Class names are inferred from subfolder names inside `dataset/train`.
- Make sure both train and test contain the same class folders.
- Your `.gitignore` is configured to keep dataset images out of Git.
- The current training script does not save the trained model yet.
