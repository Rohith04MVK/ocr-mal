import os
import time
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import random_split, DataLoader

from data.mal_dataloader import LineTextDataset, get_dataloader
from models.ocr_model import SwinTransformerOCR

def set_seed(seed=42):
    """Set seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def train():
    print("=" * 50)
    print("MALAYALAM OCR TRAINING STARTING")
    print("=" * 50)
    start_time_total = time.time()

    # --- Configuration ---
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Dataset parameters
    csv_file = "/home/rohithkb/workspace/ocr-data-gen/output/labels.csv"
    base_path = "/home/rohithkb/workspace/ocr-data-gen/output/images"
    alphabet_file = "/home/rohithkb/workspace/ocr-data-gen/output/alphabets.txt"
    img_height = 128
    img_width = 768
    batch_size = 16 # Adjust based on your GPU memory

    # Training parameters
    num_epochs = 50
    learning_rate = 1e-4
    weight_decay = 1e-4
    patience = 10  # Early stopping patience
    clip_norm = 1.0 # Gradient clipping norm

    # --- Dataset and Dataloaders ---
    print("\nLoading dataset...")
    # Use get_dataloader to create the dataset instance
    full_dataloader = get_dataloader(
        csv_file, base_path, alphabet_file,
        batch_size=batch_size, img_height=img_height, img_width=img_width,
        shuffle=True, num_workers=4, is_train=True, grayscale=False
    )
    full_dataset = full_dataloader.dataset
    
    # Get vocab size and special token indices directly from the dataset
    vocab_size = len(full_dataset.char_to_idx)
    pad_idx = full_dataset.get_pad_idx()

    # --- Model Configuration ---
    max_seq_len = 100 # Max length of the generated text sequence
    backbone_config = {
        'img_size': (img_height, img_width),
        'patch_size': 4,
        'in_chans': 3, # Using RGB images from dataloader
        'embed_dim': 96,
        'depths': [2, 2, 6, 2],
        'num_heads': [3, 6, 12, 24],
        'window_size': (4, 8),
        'drop_path_rate': 0.2,
    }
    head_config = {
        'vocab_size': vocab_size,
        'd_model': 256,
        'nhead': 8,
        'num_decoder_layers': 4,
        'dim_feedforward': 1024,
        'max_seq_length': max_seq_len,
    }

    # Log configuration parameters
    print("\nCONFIGURATION:")
    print(f"- Dataset CSV: {csv_file}")
    print(f"- Image size: {img_height}x{img_width}")
    print(f"- Batch size: {batch_size}")
    print(f"- Vocab size: {vocab_size}")
    print(f"- Learning rate: {learning_rate}")
    print(f"- Max epochs: {num_epochs}")

    # --- Create Model ---
    print("\nInitializing model...")
    model = SwinTransformerOCR(backbone_config, head_config)
    model = model.to(device)

    # --- Split Dataset ---
    # Split into train and validation
    train_size = int(0.9 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    # Create dataloaders with the correct collate function from the get_dataloader context
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, collate_fn=full_dataloader.collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, collate_fn=full_dataloader.collate_fn)

    print(f"Training on {len(train_dataset)} samples, validating on {len(val_dataset)} samples.")

    # --- Loss Function and Optimizer ---
    # Use CrossEntropyLoss, ignoring the PAD token
    criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

    # --- Training Loop ---
    best_val_loss = float('inf')
    early_stop_counter = 0

    print("\n" + "=" * 50)
    print("TRAINING STARTED")
    print("=" * 50)

    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        print(f"\nEpoch {epoch+1}/{num_epochs} - LR: {optimizer.param_groups[0]['lr']:.6f}")
        
        # Training phase
        model.train()
        train_loss = 0.0
        for i, (images, targets, target_lengths, _) in enumerate(train_loader):
            images = images.to(device)
            targets = targets.to(device) # Shape: [B, T]

            # Prepare targets for the decoder
            # Input: [PAD] token + sequence
            # Output: sequence + [EOS] token
            decoder_input = targets.clone()
            decoder_target = targets.clone()

            # Create decoder input by prepending PAD token
            pad_tokens = torch.full((images.size(0), 1), pad_idx, dtype=torch.long, device=device)
            decoder_input = torch.cat([pad_tokens, decoder_input[:, :-1]], dim=1)

            # Forward pass
            optimizer.zero_grad()
            outputs = model(images, decoder_input) # Shape: [B, T, vocab_size]

            # Calculate loss
            # Reshape for CrossEntropyLoss: [B * T, vocab_size] and [B * T]
            loss = criterion(outputs.reshape(-1, vocab_size), decoder_target.reshape(-1))
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
            optimizer.step()

            train_loss += loss.item()
            if (i + 1) % 50 == 0:
                print(f"  Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}")

        avg_train_loss = train_loss / len(train_loader)

        # Validation phase
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, targets, _, _ in val_loader:
                images = images.to(device)
                targets = targets.to(device)

                decoder_input = targets.clone()
                decoder_target = targets.clone()
                pad_tokens = torch.full((images.size(0), 1), pad_idx, dtype=torch.long, device=device)
                decoder_input = torch.cat([pad_tokens, decoder_input[:, :-1]], dim=1)

                outputs = model(images, decoder_input)
                loss = criterion(outputs.reshape(-1, vocab_size), decoder_target.reshape(-1))
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)
        scheduler.step(avg_val_loss)

        epoch_time = time.time() - epoch_start_time
        print(f"Epoch {epoch+1} Summary | Time: {epoch_time:.2f}s")
        print(f"  Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

        # Save checkpoint
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            os.makedirs('checkpoints', exist_ok=True)
            torch.save(model.state_dict(), 'checkpoints/best_model.pth')
            print(f"  ✓ Checkpoint saved! New best validation loss: {best_val_loss:.4f}")
            early_stop_counter = 0
        else:
            early_stop_counter += 1
            print(f"  ✗ No improvement for {early_stop_counter}/{patience} epochs.")

        if early_stop_counter >= patience:
            print("\nEarly stopping triggered.")
            break

    total_time = time.time() - start_time_total
    print("\n" + "=" * 50)
    print("TRAINING COMPLETED")
    print(f"Total Time: {total_time/60:.2f} minutes")
    print(f"Best Validation Loss: {best_val_loss:.4f}")
    print("=" * 50)

if __name__ == "__main__":
    set_seed(42)
    train()