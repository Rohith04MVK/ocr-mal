import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms
import re
from PIL import Image, ImageOps
import numpy as np
import io


class LineTextDataset(Dataset):
    def __init__(self, csv_file, base_path, alphabet_file, img_height=32, img_width=None, transform=None, is_train=True, grayscale=False):
        if isinstance(csv_file, pd.DataFrame):
            self.data_frame = csv_file
        else:
            self.data_frame = pd.read_csv(csv_file)
        self.base_path = base_path
        self.img_height = img_height
        self.img_width = img_width
        self.is_train = is_train
        self.grayscale = grayscale

        # Load alphabet
        if isinstance(alphabet_file, io.StringIO):
             self.alphabet = [char.strip() for char in alphabet_file.readlines() if char.strip()]
        else:
            with open(alphabet_file, 'r', encoding='utf-8') as f:
                self.alphabet = [char.strip() for char in f.readlines() if char.strip()]

        # Prepare alphabet with proper CTC blank handling
        self._prepare_alphabet()

        # Create mappings
        self.char_to_idx = {char: idx for idx, char in enumerate(self.alphabet)}
        self.idx_to_char = {idx: char for idx, char in enumerate(self.alphabet)}

        # Store blank index for CTC
        self.blank_idx = self.char_to_idx.get('[BLANK]', len(self.alphabet)) # Handle if blank is not in alphabet

        # Create tokenizer pattern for efficient parsing
        self._create_tokenizer_pattern()

        # Set to store missing characters
        self.missing_chars = set()

        # Default transforms
        if transform is None:
            self.transform = self._get_default_transforms()
        else:
            self.transform = transform

    def _get_default_transforms(self):
        """Returns a set of default transformations."""
        transform_list = []

        if self.grayscale:
            transform_list.append(transforms.Grayscale(num_output_channels=3))

        if self.is_train:
            # Add augmentations for training
            # For OCR, small geometric distortions are often helpful.
            transform_list.append(transforms.RandomAffine(
                degrees=2,          # Random rotation between -2 and 2 degrees
                translate=(0.05, 0.05), # Random horizontal/vertical shift
                scale=(0.95, 1.05), # Random zoom
                shear=2,            # Random shear transformation
                fill=255            # Fill with white for padded areas
            ))
            transform_list.append(transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1))
            transform_list.append(transforms.GaussianBlur(kernel_size=(3, 3), sigma=(0.1, 1.0)))
        
        transform_list.extend([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        return transforms.Compose(transform_list)

    def _prepare_alphabet(self):
        """Prepare alphabet with special tokens."""
        # Ensure space is in the alphabet for tokenization
        if ' ' not in self.alphabet:
            self.alphabet.append(' ')

    def _create_tokenizer_pattern(self):
        """Create a regex pattern for tokenizing text."""
        # Escape special characters in alphabet for regex
        escaped_chars = [re.escape(char) for char in self.alphabet]
        # Create a regex pattern that matches any of the alphabet characters
        self.tokenizer_pattern = re.compile('|'.join(escaped_chars))

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # Get filename and text
        filename = self.data_frame.iloc[idx]['filename']
        text = self.data_frame.iloc[idx]['text']

        # Load image
        if self.base_path:
            img_path = os.path.join(self.base_path, filename)
            image = Image.open(img_path).convert('RGB')
        else: # For in-memory testing
            image = Image.open(io.BytesIO(self.data_frame.iloc[idx]['image_bytes'])).convert('RGB')

        # --- Image Preprocessing ---

        # 1. Resize while keeping aspect ratio, but cap the width
        w, h = image.size
        aspect_ratio = w / h
        new_h = self.img_height
        new_w = int(aspect_ratio * new_h)

        if self.img_width is not None:
            new_w = min(new_w, self.img_width)

        image = image.resize((new_w, new_h), Image.LANCZOS)

        # 2. Pad with white to fixed width if necessary
        if self.img_width is not None:
            # Create a new white background image of the target size
            padded_image = Image.new('RGB', (self.img_width, self.img_height), 'white')
            # Paste the resized image onto the white background (left-aligned)
            padded_image.paste(image, (0, 0))
            image = padded_image


        # Apply transforms
        image = self.transform(image) if self.transform else image

        # Tokenize text
        tokens = self._text_to_tokens(text)

        # Convert tokens to indices
        target = torch.tensor([self.char_to_idx.get(token, self.blank_idx) for token in tokens], dtype=torch.long)

        return image, target, text

    def _text_to_tokens(self, text):
        """Convert text to tokens using the tokenizer pattern."""
        # Use regex to find all tokens
        tokens = self.tokenizer_pattern.findall(text)
        
        return tokens

    def _tokens_to_indices(self, tokens):
        """Convert a list of tokens to indices."""
        return [self.char_to_idx.get(token, self.blank_idx) for token in tokens]

    def get_blank_idx(self):
        """Return the blank token index for CTC loss"""
        return self.char_to_idx.get('[BLANK]', -1)

    def get_pad_idx(self):
        """Return the padding token index."""
        return self.char_to_idx.get('[PAD]', 0)

    def get_missing_characters(self):
        """Return the set of missing characters encountered during dataset creation."""
        return self.missing_chars

    def _add_missing_characters(self, text):
        """Add characters from the text to the missing characters set if not in alphabet."""
        for char in text:
            if char not in self.char_to_idx:
                self.missing_chars.add(char)

    def update_alphabet(self, new_alphabet_file):
        """Update the alphabet used by the dataset."""
        if isinstance(new_alphabet_file, io.StringIO):
             new_alphabet = [char.strip() for char in new_alphabet_file.readlines() if char.strip()]
        else:
            with open(new_alphabet_file, 'r', encoding='utf-8') as f:
                new_alphabet = [char.strip() for char in f.readlines() if char.strip()]

        # Update alphabet
        self.alphabet = list(set(self.alphabet + new_alphabet))

        # Recreate mappings
        self.char_to_idx = {char: idx for idx, char in enumerate(self.alphabet)}
        self.idx_to_char = {idx: char for idx, char in enumerate(self.alphabet)}

        # Update blank index for CTC
        self.blank_idx = self.char_to_idx.get('[BLANK]', len(self.alphabet))

        # Update tokenizer pattern
        self._create_tokenizer_pattern()

    def __repr__(self):
        return f"LineTextDataset(img_height={self.img_height}, img_width={self.img_width}, is_train={self.is_train})"


def collate_fn_variable_length(batch):
    """
    Custom collate function for variable length text
    """
    images, targets, texts = zip(*batch)
    
    # Get pad index from the dataset (assuming all items are from the same dataset)
    # This is a bit of a hack; ideally, the dataset object would be accessible.
    # For now, let's assume PAD index is 0, or get it from a passed-in dataset object.
    pad_idx = 0 # A common default. Update if your [PAD] token has a different index.

    # Pad targets to the same length
    padded_targets = torch.nn.utils.rnn.pad_sequence(targets, batch_first=True, padding_value=pad_idx)

    # Get sequence lengths (original lengths before padding)
    lengths = torch.IntTensor([len(t) for t in targets])

    # Stack images
    images = torch.stack(images, 0)

    return images, padded_targets, lengths, texts


def get_dataloader(csv_file, base_path, alphabet_file, batch_size=32, img_height=32, img_width=None,
                   shuffle=True, num_workers=4, is_train=True, grayscale=False):
    """
    Create a dataloader for the line text dataset
    """
    dataset = LineTextDataset(csv_file, base_path, alphabet_file, img_height, img_width, is_train=is_train, grayscale=grayscale)

    # Update collate_fn to use the correct padding index from the dataset
    def collate_fn_with_padding(batch):
        images, targets, texts = zip(*batch)
        pad_idx = dataset.get_pad_idx()
        padded_targets = torch.nn.utils.rnn.pad_sequence(targets, batch_first=True, padding_value=pad_idx)
        lengths = torch.IntTensor([len(t) for t in targets])
        images = torch.stack(images, 0)
        return images, padded_targets, lengths, texts

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn_with_padding
    )

    return dataloader


if __name__ == "__main__":
    # This test assumes you have a dataset structured as follows:
    # - /path/to/your/dataset/
    #   - images/
    #     - image1.png
    #     - ...
    #   - labels.csv
    #   - alphabet.txt

    print("Testing LineTextDataset and DataLoader with an existing dataset...")

    # --- 1. Point to your existing dataset directory ---
    test_dir = "/home/rohithkb/workspace/ocr-data-gen/output/"
    images_dir = os.path.join(test_dir, "images")
    print(images_dir)
    csv_path = os.path.join(test_dir, "labels.csv")
    alphabet_path = os.path.join(test_dir, "alphabets.txt")

    # Check if the paths exist
    if not all(os.path.exists(p) for p in [images_dir, csv_path, alphabet_path]):
        print("Error: Dataset not found at the specified path.")
        print("Please ensure the following exist:")
        print(f" - Image directory: {images_dir}")
        print(f" - CSV file: {csv_path}")
        print(f" - Alphabet file: {alphabet_path}")
    else:
        # --- 2. Test the Dataset ---
        IMG_HEIGHT, IMG_WIDTH = 128, 512
        dataset = LineTextDataset(
            csv_file=csv_path,
            base_path=images_dir,
            alphabet_file=alphabet_path,
            img_height=IMG_HEIGHT,
            img_width=IMG_WIDTH,
            is_train=True,
            grayscale=True
        )
        
        print(f"Dataset size: {len(dataset)}")
        print(f"Alphabet size: {len(dataset.alphabet)}")
        print(f"Char to Idx mapping sample: {list(dataset.char_to_idx.items())[:5]}")
        print(f"PAD index: {dataset.get_pad_idx()}")

        # --- 3. Test a single item ---
        image, target, text = dataset[0]
        print("\n--- Single Item Test ---")
        print(f"Original text: '{text}'")
        print(f"Image shape: {image.shape}")
        print(f"Target tensor: {target}")
        print(f"Target shape: {target.shape}")
        assert image.shape == (3, IMG_HEIGHT, IMG_WIDTH)

        # --- 4. Test the DataLoader ---
        dataloader = get_dataloader(
            csv_file=csv_path,
            base_path=images_dir,
            alphabet_file=alphabet_path,
            batch_size=4,
            img_height=IMG_HEIGHT,
            img_width=IMG_WIDTH,
            shuffle=False,
            num_workers=0, # Use 0 for main process testing
            is_train=True,
            grayscale=True
        )

        images, padded_targets, lengths, texts = next(iter(dataloader))
        print("\n--- DataLoader Batch Test ---")
        print(f"Images batch shape: {images.shape}")
        print(f"Padded targets batch shape: {padded_targets.shape}")
        print(f"Sequence lengths: {lengths}")
        print(f"Texts in batch: {texts}")
        assert images.shape == (4, 3, IMG_HEIGHT, IMG_WIDTH)
        assert padded_targets.shape[0] == 4

        # --- 5. Test Tokenization on a single sentence ---
        print("\n--- Tokenization Test ---")
        # V-- REPLACE THIS WITH YOUR SENTENCE --V
        test_sentence = "നടതുറക്കുന്നത് നടതുറക്കുന്നത്"
        print(f"Original sentence: '{test_sentence}'")
        
        # Tokenize the sentence using the dataset's method
        tokens = dataset._text_to_tokens(test_sentence)
        print(f"Tokens: {tokens}")

        # Convert tokens to indices
        indices = dataset._tokens_to_indices(tokens)
        print(f"Indices: {indices}")

        print("\nAll tests passed!")