import os
import argparse
import torch
from PIL import Image
import torchvision.transforms as transforms
import io

from models.ocr_model import SwinTransformerOCR
from data.mal_dataloader import LineTextDataset


def load_model(checkpoint_path, alphabet_file, model_config, device):
    """Load trained model from checkpoint"""
    # Create a dummy dataset to get vocab mappings
    dummy_data = io.StringIO("dummy,dummy\n")
    temp_dataset = LineTextDataset(
        csv_file=dummy_data,
        base_path=None,
        alphabet_file=alphabet_file,
        img_height=model_config['img_height'],
        img_width=model_config['img_width'],
        grayscale=model_config['grayscale']
    )
    
    vocab_size = len(temp_dataset.char_to_idx)
    
    # Model configuration
    backbone_config = {
        'img_size': (model_config['img_height'], model_config['img_width']),
        'patch_size': 4,
        'in_chans': 1 if model_config['grayscale'] else 3,
        'embed_dim': model_config['embed_dim'],
        'depths': [2, 2, 6, 2],
        'num_heads': [3, 6, 12, 24],
        'window_size': (4, 8),
        'drop_path_rate': 0.2,
    }
    
    head_config = {
        'vocab_size': vocab_size,
        'd_model': model_config['d_model'],
        'nhead': model_config['nhead'],
        'num_decoder_layers': model_config['num_decoder_layers'],
        'dim_feedforward': model_config['d_model'] * 4,
        'max_seq_length': model_config['max_seq_len'],
    }
    
    # Create and load model
    model = SwinTransformerOCR(backbone_config, head_config)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model = model.to(device)
    model.eval()
    
    return model, temp_dataset


def preprocess_image(image_path, img_height, img_width, grayscale=False):
    """Preprocess image for inference"""
    # Load image
    image = Image.open(image_path).convert('RGB')
    
    # Resize while keeping aspect ratio
    w, h = image.size
    aspect_ratio = w / h
    new_h = img_height
    new_w = int(aspect_ratio * new_h)
    
    if img_width is not None:
        new_w = min(new_w, img_width)
    
    image = image.resize((new_w, new_h), Image.LANCZOS)
    
    # Pad with white to fixed width if necessary
    if img_width is not None:
        padded_image = Image.new('RGB', (img_width, img_height), 'white')
        padded_image.paste(image, (0, 0))
        image = padded_image
    
    # Apply transforms
    transform_list = []
    
    if grayscale:
        transform_list.append(transforms.Grayscale(num_output_channels=3))
    
    transform_list.extend([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    transform = transforms.Compose(transform_list)
    image = transform(image).unsqueeze(0)  # Add batch dimension
    
    return image


def decode_prediction(prediction_indices, dataset):
    """Convert prediction indices to text"""
    text = []
    pad_idx = dataset.get_pad_idx()
    
    for idx in prediction_indices:
        if idx == pad_idx:
            break
        if idx in dataset.idx_to_char:
            char = dataset.idx_to_char[idx]
            if char not in ['[SOS]', '[EOS]', '[PAD]', '[BLANK]']:
                text.append(char)
    
    return ''.join(text)


def predict_single_image(model, image_path, dataset, model_config, device, 
                        sos_token_idx=None, eos_token_idx=None):
    """Predict text for a single image"""
    # Get SOS and EOS indices
    if sos_token_idx is None:
        sos_token_idx = dataset.char_to_idx.get('[SOS]', 1)
    if eos_token_idx is None:
        eos_token_idx = dataset.char_to_idx.get('[EOS]', 2)
    
    # Preprocess image
    image = preprocess_image(
        image_path, 
        model_config['img_height'], 
        model_config['img_width'],
        model_config['grayscale']
    ).to(device)
    
    # Generate prediction
    with torch.no_grad():
        prediction = model.predict(image, sos_token_idx, eos_token_idx)
    
    # Decode prediction
    predicted_text = decode_prediction(prediction[0].cpu().numpy(), dataset)
    
    return predicted_text


def predict_batch_images(model, image_paths, dataset, model_config, device,
                        sos_token_idx=None, eos_token_idx=None):
    """Predict text for a batch of images"""
    # Get SOS and EOS indices
    if sos_token_idx is None:
        sos_token_idx = dataset.char_to_idx.get('[SOS]', 1)
    if eos_token_idx is None:
        eos_token_idx = dataset.char_to_idx.get('[EOS]', 2)
    
    # Preprocess all images
    images = []
    for image_path in image_paths:
        image = preprocess_image(
            image_path,
            model_config['img_height'],
            model_config['img_width'],
            model_config['grayscale']
        )
        images.append(image)
    
    # Stack images into batch
    batch_images = torch.cat(images, dim=0).to(device)
    
    # Generate predictions
    with torch.no_grad():
        predictions = model.predict(batch_images, sos_token_idx, eos_token_idx)
    
    # Decode predictions
    predicted_texts = []
    for i in range(predictions.size(0)):
        predicted_text = decode_prediction(predictions[i].cpu().numpy(), dataset)
        predicted_texts.append(predicted_text)
    
    return predicted_texts


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Inference for Malayalam OCR model')
    
    # Model and data arguments
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint (.pth file)')
    parser.add_argument('--alphabet-file', type=str, required=True,
                       help='Path to alphabet file used during training')
    
    # Image arguments
    parser.add_argument('--image', type=str, default=None,
                       help='Path to single image for prediction')
    parser.add_argument('--image-dir', type=str, default=None,
                       help='Directory containing images for batch prediction')
    parser.add_argument('--output', type=str, default=None,
                       help='Output file to save predictions (optional)')
    
    # Model configuration (should match training config)
    parser.add_argument('--img-height', type=int, default=128,
                       help='Image height (should match training config)')
    parser.add_argument('--img-width', type=int, default=768,
                       help='Image width (should match training config)')
    parser.add_argument('--grayscale', action='store_true', default=False,
                       help='Use grayscale images (should match training config)')
    parser.add_argument('--embed-dim', type=int, default=96,
                       help='Embedding dimension (should match training config)')
    parser.add_argument('--d-model', type=int, default=256,
                       help='Transformer model dimension (should match training config)')
    parser.add_argument('--nhead', type=int, default=8,
                       help='Number of attention heads (should match training config)')
    parser.add_argument('--num-decoder-layers', type=int, default=4,
                       help='Number of decoder layers (should match training config)')
    parser.add_argument('--max-seq-len', type=int, default=100,
                       help='Maximum sequence length (should match training config)')
    
    # Other arguments
    parser.add_argument('--batch-size', type=int, default=8,
                       help='Batch size for batch prediction')
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Validate arguments
    if not args.image and not args.image_dir:
        raise ValueError("Either --image or --image-dir must be provided")
    
    if args.image and args.image_dir:
        raise ValueError("Only one of --image or --image-dir should be provided")
    
    # Device setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Model configuration
    model_config = {
        'img_height': args.img_height,
        'img_width': args.img_width,
        'grayscale': args.grayscale,
        'embed_dim': args.embed_dim,
        'd_model': args.d_model,
        'nhead': args.nhead,
        'num_decoder_layers': args.num_decoder_layers,
        'max_seq_len': args.max_seq_len,
    }
    
    print("Loading model...")
    model, dataset = load_model(args.checkpoint, args.alphabet_file, model_config, device)
    print(f"Model loaded successfully. Vocab size: {len(dataset.char_to_idx)}")
    
    # Single image prediction
    if args.image:
        print(f"Predicting text for image: {args.image}")
        predicted_text = predict_single_image(
            model, args.image, dataset, model_config, device
        )
        print(f"Predicted text: '{predicted_text}'")
        
        if args.output:
            with open(args.output, 'w', encoding='utf-8') as f:
                f.write(f"{args.image}\t{predicted_text}\n")
            print(f"Prediction saved to: {args.output}")
    
    # Batch prediction
    elif args.image_dir:
        print(f"Processing images in directory: {args.image_dir}")
        
        # Get all image files
        image_extensions = ['.png', '.jpg', '.jpeg', '.bmp', '.tiff']
        image_files = []
        for file in os.listdir(args.image_dir):
            if any(file.lower().endswith(ext) for ext in image_extensions):
                image_files.append(os.path.join(args.image_dir, file))
        
        image_files.sort()
        print(f"Found {len(image_files)} images")
        
        # Process in batches
        all_predictions = []
        for i in range(0, len(image_files), args.batch_size):
            batch_files = image_files[i:i + args.batch_size]
            print(f"Processing batch {i//args.batch_size + 1}/{(len(image_files)-1)//args.batch_size + 1}")
            
            predictions = predict_batch_images(
                model, batch_files, dataset, model_config, device
            )
            
            for file_path, prediction in zip(batch_files, predictions):
                filename = os.path.basename(file_path)
                print(f"  {filename}: '{prediction}'")
                all_predictions.append((file_path, prediction))
        
        # Save predictions if output file specified
        if args.output:
            with open(args.output, 'w', encoding='utf-8') as f:
                for file_path, prediction in all_predictions:
                    f.write(f"{os.path.basename(file_path)}\t{prediction}\n")
            print(f"All predictions saved to: {args.output}")
    
    print("Inference completed!")


if __name__ == "__main__":
    main()