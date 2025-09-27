import torch
import editdistance

def calculate_cer(predictions, targets, dataset):
    """
    Calculate Character Error Rate (CER) for OCR predictions.
    
    Args:
        predictions (torch.Tensor): Predicted token indices [B, T]
        targets (torch.Tensor): Target token indices [B, T]
        dataset: Dataset object with idx_to_char mapping
        
    Returns:
        float: Average CER across the batch
    """
    batch_size = predictions.size(0)
    total_cer = 0.0
    total_chars = 0
    
    pad_idx = dataset.get_pad_idx()
    
    for i in range(batch_size):
        # Convert predictions to text (remove padding and stop at first pad token)
        pred_indices = predictions[i].cpu().numpy()
        target_indices = targets[i].cpu().numpy()
        
        # Remove padding tokens and convert to text
        pred_text = indices_to_text(pred_indices, dataset.idx_to_char, pad_idx)
        target_text = indices_to_text(target_indices, dataset.idx_to_char, pad_idx)
        
        # Calculate edit distance
        if len(target_text) > 0:
            edit_dist = editdistance.eval(pred_text, target_text)
            cer = edit_dist / len(target_text)
            total_cer += cer
            total_chars += len(target_text)
        
    return total_cer / batch_size if batch_size > 0 else 0.0

def indices_to_text(indices, idx_to_char, pad_idx):
    """
    Convert token indices to text string.
    
    Args:
        indices: numpy array of token indices
        idx_to_char: dictionary mapping indices to characters
        pad_idx: padding token index
        
    Returns:
        str: Decoded text string
    """
    text = []
    for idx in indices:
        if idx == pad_idx:
            break
        if idx in idx_to_char:
            text.append(idx_to_char[idx])
    
    return ''.join(text)