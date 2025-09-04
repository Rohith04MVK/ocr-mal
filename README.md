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

All training parameters are set directly in [`train.py`](train.py). You must update the dataset paths before running.

```python
// filepath: train.py
// ...existing code...
    # Dataset parameters
    csv_file = "/path/to/your/labels.csv"
    base_path = "/path/to/your/images"
    alphabet_file = "/path/to/your/alphabets.txt"
// ...existing code...
```

Once configured, start training:

```sh
python train.py
```

The script will train the model, validate it after each epoch, and save the best-performing model to `checkpoints/best_model.pth`.

Note: This project is not configured with command-line arguments. It was a quick setup to test the architecture, so all changes need to be made directly in the source files.