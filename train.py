import os
import time
import random
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import random_split, DataLoader

from data.mal_dataloader import LineTextDataset, get_dataloader
from models.ocr_model import SwinTransformerOCR
from utils.metrics import calculate_cer

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

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Train Malayalam OCR model')
    
    # Dataset arguments
    parser.add_argument('--csv-file', type=str, default='labels.csv',
                       help='Path to CSV file containing image-text pairs (default: labels.csv)')
    parser.add_argument('--base-path', type=str, default='images',
                       help='Base path to images directory (default: images)')
    parser.add_argument('--alphabet-file', type=str, default='alphabets.txt',
                       help='Path to alphabet file (default: alphabets.txt)')
    
    # Image processing arguments
    parser.add_argument('--img-height', type=int, default=128,
                       help='Image height for resizing (default: 128)')
    parser.add_argument('--img-width', type=int, default=768,
                       help='Image width for resizing (default: 768)')
    parser.add_argument('--grayscale', action='store_true', default=False,
                       help='Use grayscale images instead of RGB (default: False)')
    
    # Training arguments
    parser.add_argument('--batch-size', type=int, default=16,
                       help='Training batch size (default: 16)')
    parser.add_argument('--num-epochs', type=int, default=50,
                       help='Number of training epochs (default: 50)')
    parser.add_argument('--learning-rate', type=float, default=1e-4,
                       help='Learning rate (default: 1e-4)')
    parser.add_argument('--weight-decay', type=float, default=1e-4,
                       help='Weight decay (default: 1e-4)')
    parser.add_argument('--patience', type=int, default=10,
                       help='Early stopping patience (default: 10)')
    parser.add_argument('--clip-norm', type=float, default=1.0,
                       help='Gradient clipping norm (default: 1.0)')
    parser.add_argument('--max-seq-len', type=int, default=100,
                       help='Maximum sequence length (default: 100)')
    
    # Data split arguments
    parser.add_argument('--train-split', type=float, default=0.9,
                       help='Training set split ratio (default: 0.9)')
    parser.add_argument('--num-workers', type=int, default=4,
                       help='Number of dataloader workers (default: 4)')
    
    # Model arguments
    parser.add_argument('--embed-dim', type=int, default=96,
                       help='Swin transformer embedding dimension (default: 96)')
    parser.add_argument('--d-model', type=int, default=256,
                       help='Transformer decoder model dimension (default: 256)')
    parser.add_argument('--nhead', type=int, default=8,
                       help='Number of attention heads (default: 8)')
    parser.add_argument('--num-decoder-layers', type=int, default=4,
                       help='Number of decoder layers (default: 4)')
    
    # Other arguments
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed (default: 42)')
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints',
                       help='Directory to save checkpoints (default: checkpoints)')
    
    return parser.parse_args()

def train(args):
    print("=" * 50)
    print("MALAYALAM OCR TRAINING STARTING")
    print("=" * 50)
    start_time_total = time.time()

    # --- Configuration ---
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # --- Dataset and Dataloaders ---
    print("\nLoading dataset...")
    # Use get_dataloader to create the dataset instance
    full_dataloader = get_dataloader(
        args.csv_file, args.base_path, args.alphabet_file,
        batch_size=args.batch_size, img_height=args.img_height, img_width=args.img_width,
        shuffle=True, num_workers=args.num_workers, is_train=True, grayscale=args.grayscale
    )
    full_dataset = full_dataloader.dataset
    
    # Get vocab size and special token indices directly from the dataset
    vocab_size = len(full_dataset.char_to_idx)
    pad_idx = full_dataset.get_pad_idx()

    # --- Model Configuration ---
    backbone_config = {
        'img_size': (args.img_height, args.img_width),
        'patch_size': 4,
        'in_chans': 1 if args.grayscale else 3,
        'embed_dim': args.embed_dim,
        'depths': [2, 2, 6, 2],
        'num_heads': [3, 6, 12, 24],
        'window_size': (4, 8),
        'drop_path_rate': 0.2,
    }
    head_config = {
        'vocab_size': vocab_size,
        'd_model': args.d_model,
        'nhead': args.nhead,
        'num_decoder_layers': args.num_decoder_layers,
        'dim_feedforward': args.d_model * 4,
        'max_seq_length': args.max_seq_len,
    }

    # Log configuration parameters
    print("\nCONFIGURATION:")
    print(f"- Dataset CSV: {args.csv_file}")
    print(f"- Images path: {args.base_path}")
    print(f"- Alphabet file: {args.alphabet_file}")
    print(f"- Image size: {args.img_height}x{args.img_width}")
    print(f"- Grayscale: {args.grayscale}")
    print(f"- Batch size: {args.batch_size}")
    print(f"- Vocab size: {vocab_size}")
    print(f"- Learning rate: {args.learning_rate}")
    print(f"- Max epochs: {args.num_epochs}")
    print(f"- Max sequence length: {args.max_seq_len}")

    # --- Create Model ---
    print("\nInitializing model...")
    model = SwinTransformerOCR(backbone_config, head_config)
    model = model.to(device)

    # --- Split Dataset ---
    # Split into train and validation
    train_size = int(args.train_split * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    # Create dataloaders with the correct collate function from the get_dataloader context
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, 
                             num_workers=args.num_workers, collate_fn=full_dataloader.collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, 
                           num_workers=args.num_workers, collate_fn=full_dataloader.collate_fn)

    print(f"Training on {len(train_dataset)} samples, validating on {len(val_dataset)} samples.")

    # --- Loss Function and Optimizer ---
    # Use CrossEntropyLoss, ignoring the PAD token
    criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

    # --- Training Loop ---
    best_val_loss = float('inf')
    early_stop_counter = 0

    print("\n" + "=" * 50)
    print("TRAINING STARTED")
    print("=" * 50)

    for epoch in range(args.num_epochs):
        epoch_start_time = time.time()
        print(f"\nEpoch {epoch+1}/{args.num_epochs} - LR: {optimizer.param_groups[0]['lr']:.6f}")
        
        # Training phase
        model.train()
        train_loss = 0.0
        for i, (images, targets, target_lengths, _) in enumerate(train_loader):
            images = images.to(device)
            targets = targets.to(device)

            decoder_input = targets.clone()
            decoder_target = targets.clone()

            pad_tokens = torch.full((images.size(0), 1), pad_idx, dtype=torch.long, device=device)
            decoder_input = torch.cat([pad_tokens, decoder_input[:, :-1]], dim=1)

            optimizer.zero_grad()
            outputs = model(images, decoder_input)

            loss = criterion(outputs.reshape(-1, vocab_size), decoder_target.reshape(-1))
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_norm)
            optimizer.step()

            train_loss += loss.item()
            if (i + 1) % 50 == 0:
                print(f"  Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}")

        avg_train_loss = train_loss / len(train_loader)

        # Validation phase
        model.eval()
        val_loss = 0.0
        total_cer = 0.0
        num_val_batches = 0
        
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
                
                # Calculate CER
                # Get predicted tokens (argmax of logits)
                predictions = torch.argmax(outputs, dim=-1)  # [B, T]
                
                # Calculate CER for this batch
                batch_cer = calculate_cer(predictions, decoder_target, full_dataset)
                total_cer += batch_cer
                num_val_batches += 1

        avg_val_loss = val_loss / len(val_loader)
        avg_cer = total_cer / num_val_batches if num_val_batches > 0 else 0.0
        
        scheduler.step(avg_val_loss)

        epoch_time = time.time() - epoch_start_time
        print(f"Epoch {epoch+1} Summary | Time: {epoch_time:.2f}s")
        print(f"  Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val CER: {avg_cer:.4f}")

        # Save checkpoint
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            os.makedirs(args.checkpoint_dir, exist_ok=True)
            torch.save(model.state_dict(), os.path.join(args.checkpoint_dir, 'best_model.pth'))
            print(f"  ✓ Checkpoint saved! New best validation loss: {best_val_loss:.4f}, CER: {avg_cer:.4f}")
            early_stop_counter = 0
        else:
            early_stop_counter += 1
            print(f"  ✗ No improvement for {early_stop_counter}/{args.patience} epochs.")

        if early_stop_counter >= args.patience:
            print("\nEarly stopping triggered.")
            break

    total_time = time.time() - start_time_total
    print("\n" + "=" * 50)
    print("TRAINING COMPLETED")
    print(f"Total Time: {total_time/60:.2f} minutes")
    print(f"Best Validation Loss: {best_val_loss:.4f}")
    print("=" * 50)

if __name__ == "__main__":
    args = parse_args()
    set_seed(args.seed)
    train(args)