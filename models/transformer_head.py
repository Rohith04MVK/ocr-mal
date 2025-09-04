import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    """Injects some information about the relative or absolute position of the tokens in the sequence."""
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class TransformerHead(nn.Module):
    """
    A Transformer-based decoder head for OCR.
    It takes features from an encoder (like a SwinTransformer) and generates a sequence of characters.
    """
    def __init__(self, input_dim, vocab_size, d_model=512, nhead=8, num_decoder_layers=6,
                 dim_feedforward=2048, max_seq_length=100, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.max_seq_length = max_seq_length

        self.input_proj = nn.Conv2d(input_dim, d_model, kernel_size=1)
        
        self.pos_encoder = PositionalEncoding(d_model, dropout, max_len=max_seq_length)
        self.tgt_embedding = nn.Embedding(vocab_size, d_model)
        
        decoder_layer = nn.TransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout, activation='relu')
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_decoder_layers)
        
        self.fc_out = nn.Linear(d_model, vocab_size)

        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def generate_square_subsequent_mask(self, sz):
        """Generates a square mask for the sequence. The masked positions are filled with float('-inf')."""
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, src, tgt):
        """
        Forward pass for the TransformerHead.
        Args:
            src (Tensor): Features from the backbone encoder. Shape: [B, C, H, W]
            tgt (Tensor): The target sequence. Shape: [B, T] where T is target sequence length.
        """
        # Project input features to d_model
        src = self.input_proj(src) # [B, d_model, H, W]
        
        # Reshape and add positional encoding for the source
        B, C, H, W = src.shape
        src = src.flatten(2).permute(2, 0, 1) # [H*W, B, d_model]
        
        # Embed target sequence and add positional encoding
        tgt = tgt.transpose(0, 1) # [T, B]
        tgt_embed = self.tgt_embedding(tgt) * math.sqrt(self.d_model)
        tgt_embed = self.pos_encoder(tgt_embed) # [T, B, d_model]

        # Generate mask for the target sequence
        tgt_mask = self.generate_square_subsequent_mask(tgt.size(0)).to(src.device)

        # Decode
        output = self.transformer_decoder(tgt_embed, src, tgt_mask=tgt_mask) # [T, B, d_model]
        
        # Final output layer
        output = self.fc_out(output) # [T, B, vocab_size]
        
        return output.permute(1, 0, 2) # [B, T, vocab_size]

if __name__ == '__main__':
    # Small test to verify the model
    
    # Parameters from the Swin backbone output
    B, C, H, W = 2, 768, 4, 16
    
    # Vocabulary and sequence length
    VOCAB_SIZE = 100
    MAX_SEQ_LEN = 50
    
    # Create a Transformer Head
    head = TransformerHead(
        input_dim=C,
        vocab_size=VOCAB_SIZE,
        d_model=256,
        nhead=8,
        num_decoder_layers=3,
        max_seq_length=MAX_SEQ_LEN
    )
    # Create dummy input tensors
    # This simulates the output from the SwinTransformerBackbone
    encoder_output = torch.randn(B, C, H, W)
    
    # This simulates a batch of target sequences (e.g., during training)
    # Target length is 30 for this example
    target_sequence = torch.randint(1, VOCAB_SIZE, (B, 30))

    # Get the output from the model
    output = head(encoder_output, target_sequence)

    # Print the shapes
    print(f"\nTransformerHead test")
    print(f"Encoder output shape: {encoder_output.shape}")
    print(f"Target sequence shape: {target_sequence.shape}")
    print(f"Final output shape: {output.shape}")

    # Expected shape: (Batch, Target Length, Vocab Size)
    expected_shape = (B, target_sequence.size(1), VOCAB_SIZE)
    assert output.shape == expected_shape, f"Shape mismatch! Expected {expected_shape}, got {output.shape}"
    print("Test passed!")