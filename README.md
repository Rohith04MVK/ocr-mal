# ocr-mal

An end-to-end OCR model for Malayalam text. This was built to experiment with a Swin Transformer backbone and a standard Transformer decoder for a sequence-to-sequence task.

### Architecture

The model is composed of two main parts:

1.  **Backbone**: A [`SwinTransformerBackbone`](models/swin_backbone.py) (from `timm`) acts as the feature extractor, processing the input image and producing a feature map.
2.  **Head**: A [`TransformerHead`](models/transformer_head.py) takes the feature map from the backbone and uses a decoder to generate the output text sequence.

These two components are combined in the [`SwinTransformerOCR`](models/ocr_model.py) model.

### Project Structure

-   `train.py`: The main script for training the model. All configurations are hardcoded here.
-   `data/mal_dataloader.py`: Contains the `LineTextDataset` class and dataloader for loading and preprocessing images and text.
-   `models/`: Contains the model definition files.
    -   `swin_backbone.py`: The Swin Transformer encoder.
    -   `transformer_head.py`: The Transformer decoder.
    -   `ocr_model.py`: The complete end-to-end model.
-   `checkpoints/`: This directory will be created to save the best model weights during training.

### Setup

You'll need Python 3.11. Create a virtual environment and install the dependencies.

```sh
# using uv
uv venv
source .venv/bin/activate
uv pip install torch torchvision timm pandas Pillow numpy
```

### Usage

The training script now supports command-line arguments for easy configuration:

```sh
# Basic usage with default parameters
python train.py

# Custom dataset paths
python train.py --csv-file /path/to/labels.csv --base-path /path/to/images --alphabet-file /path/to/alphabets.txt

# Custom training parameters
python train.py --batch-size 32 --learning-rate 2e-4 --num-epochs 100

# Use grayscale images
python train.py --grayscale

# Full example with custom settings
python train.py \
    --csv-file data/my_labels.csv \
    --base-path data/my_images \
    --alphabet-file data/my_alphabets.txt \
    --batch-size 32 \
    --img-height 64 \
    --img-width 512 \
    --num-epochs 100 \
    --learning-rate 2e-4
```

#### Available Arguments:

**Dataset Arguments:**
- `--csv-file`: Path to CSV file containing image-text pairs (default: `labels.csv`)
- `--base-path`: Base path to images directory (default: `images`)
- `--alphabet-file`: Path to alphabet file (default: `alphabets.txt`)

**Image Processing:**
- `--img-height`: Image height for resizing (default: `128`)
- `--img-width`: Image width for resizing (default: `768`)
- `--grayscale`: Use grayscale images instead of RGB

**Training Parameters:**
- `--batch-size`: Training batch size (default: `16`)
- `--num-epochs`: Number of training epochs (default: `50`)
- `--learning-rate`: Learning rate (default: `1e-4`)
- `--weight-decay`: Weight decay (default: `1e-4`)
- `--patience`: Early stopping patience (default: `10`)

**Model Parameters:**
- `--embed-dim`: Swin transformer embedding dimension (default: `96`)
- `--d-model`: Transformer decoder model dimension (default: `256`)
- `--max-seq-len`: Maximum sequence length (default: `100`)

Use `python train.py --help` to see all available options.

Once configured, start training:

```sh
python train.py
```

The script will train the model, validate it after each epoch, and save the best-performing model to `checkpoints/best_model.pth`.

Note: This project is not configured with command-line arguments. It was a quick setup to test the architecture, so all changes need to be made directly in the source files.