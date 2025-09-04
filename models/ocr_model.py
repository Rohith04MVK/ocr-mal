import torch
import torch.nn as nn

from models.swin_backbone import SwinTransformerBackbone
from models.transformer_head import TransformerHead

class SwinTransformerOCR(nn.Module):
    """
    A complete OCR model combining a Swin Transformer backbone and a Transformer decoder head.
    """
    def __init__(self, backbone_config, head_config):
        """
        Args:
            backbone_config (dict): Configuration for the SwinTransformerBackbone.
            head_config (dict): Configuration for the TransformerHead.
        """
        super().__init__()
        self.backbone = SwinTransformerBackbone(**backbone_config)
        
        # The input_dim for the head must match the output channels of the backbone.
        # This is automatically determined.
        backbone_out_channels = self.backbone.num_features
        head_config['input_dim'] = backbone_out_channels
        
        self.head = TransformerHead(**head_config)

    def forward(self, image, target):
        """
        Forward pass for the full OCR model.
        
        Args:
            image (Tensor): Input image tensor. Shape: [B, C, H, W]
            target (Tensor): The target sequence for teacher forcing. Shape: [B, T]
        
        Returns:
            Tensor: The output logits from the decoder. Shape: [B, T, vocab_size]
        """
        features = self.backbone(image)
        output = self.head(features, target)
        return output

    @torch.no_grad()
    def predict(self, image, sos_token_idx=1, eos_token_idx=2):
        """
        Generates predictions for a batch of images using greedy decoding.

        Args:
            image (Tensor): Input image tensor. Shape: [B, C, H, W]
            sos_token_idx (int): The index for the start-of-sequence token.
            eos_token_idx (int): The index for the end-of-sequence token.

        Returns:
            Tensor: The predicted sequence of token indices. Shape: [B, T]
        """
        self.eval()
        features = self.backbone(image)
        
        B = image.size(0)
        max_len = self.head.max_seq_length
        
        # Start with the SOS token for each image in the batch
        predicted_sequence = torch.full((B, 1), sos_token_idx, dtype=torch.long, device=image.device)
        
        for _ in range(max_len - 1):
            # Get the model output for the current sequence
            output_logits = self.head(features, predicted_sequence)
            
            # Get the logits for the last token and find the most likely next token
            last_token_logits = output_logits[:, -1, :]
            next_token = torch.argmax(last_token_logits, dim=-1).unsqueeze(1)
            
            # Append the new token to the sequence
            predicted_sequence = torch.cat([predicted_sequence, next_token], dim=1)
            
            # Stop if all sequences in the batch have generated an EOS token
            if (next_token.squeeze() == eos_token_idx).all():
                break
                
        return predicted_sequence

if __name__ == '__main__':
    # Small test to verify the combined model
    
    # Configuration
    IMG_SIZE = (128, 512)
    VOCAB_SIZE = 100
    MAX_SEQ_LEN = 50
    
    backbone_config = {
        'img_size': IMG_SIZE,
        'patch_size': 4,
        'in_chans': 1,
        'embed_dim': 96,
        'depths': [2, 2, 6, 2],
        'num_heads': [3, 6, 12, 24],
        'window_size': (4, 8), # Adjusted for non-square images
        'drop_path_rate': 0.2,
    }
    
    head_config = {
        'vocab_size': VOCAB_SIZE,
        'd_model': 256,
        'nhead': 8,
        'num_decoder_layers': 3,
        'max_seq_length': MAX_SEQ_LEN,
    }
    
    # Create the model
    model = SwinTransformerOCR(backbone_config, head_config)
    
    # Create dummy inputs
    B = 2
    dummy_image = torch.randn(B, 1, IMG_SIZE[0], IMG_SIZE[1])
    dummy_target = torch.randint(1, VOCAB_SIZE, (B, 30)) # Target length = 30
    
    # Forward pass
    output = model(dummy_image, dummy_target)
    
    # Print shapes
    print("\nSwinTransformerOCR test")
    print(f"Image input shape: {dummy_image.shape}")
    print(f"Target input shape: {dummy_target.shape}")
    print(f"Final output shape: {output.shape}")
    
    # Expected shape: (Batch, Target Length, Vocab Size)
    expected_shape = (B, dummy_target.size(1), VOCAB_SIZE)
    assert output.shape == expected_shape, f"Shape mismatch! Expected {expected_shape}, got {output.shape}"
    print("Test passed!")

    # --- Test for prediction/inference ---
    print("\nSwinTransformerOCR prediction test")
    
    # Assume SOS=1, EOS=2
    SOS_TOKEN = 1
    EOS_TOKEN = 2
    
    # Use the first image from the dummy batch for prediction
    dummy_image_for_pred = dummy_image[0].unsqueeze(0)
    
    # Generate prediction
    predicted_sequence = model.predict(dummy_image_for_pred, sos_token_idx=SOS_TOKEN, eos_token_idx=EOS_TOKEN)
    
    print(f"Image input shape for prediction: {dummy_image_for_pred.shape}")
    print(f"Predicted sequence of token IDs: {predicted_sequence}")
    print(f"Predicted sequence shape: {predicted_sequence.shape}")
    print("Prediction test passed!")