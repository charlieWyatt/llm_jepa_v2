import torch
import torch.nn as nn
import torch.nn.functional as F


class NTPLoss(nn.Module):
    """Next Token Prediction loss for causal language modeling."""
    
    def __init__(self, hidden_size: int, vocab_size: int, pad_token_id: int = 0):
        super().__init__()
        self.pad_token_id = pad_token_id
        self.vocab_size = vocab_size
        # LM head will be set externally or created here
        self.lm_head = None
        
    def set_lm_head(self, lm_head: nn.Module):
        """Set LM head from existing model."""
        self.lm_head = lm_head
        
    def forward(
        self,
        hidden_states: torch.Tensor,  # [B, L, H]
        input_ids: torch.Tensor,       # [B, L]
        attention_mask: torch.Tensor = None,  # [B, L]
    ) -> torch.Tensor:
        """
        Compute NTP loss.
        
        Args:
            hidden_states: Model hidden states [B, L, H]
            input_ids: Input token IDs [B, L]
            attention_mask: Optional attention mask [B, L]
        
        Returns:
            NTP loss scalar
        """
        if self.lm_head is None:
            raise ValueError("LM head not set. Call set_lm_head() first.")
        
        # Get logits
        lm_logits = self.lm_head(hidden_states)  # [B, L, V]
        
        # Shift for causal prediction: logits[:-1] predicts tokens[1:]
        shift_logits = lm_logits[:, :-1, :].contiguous()
        shift_labels = input_ids[:, 1:].contiguous()
        
        # Compute cross entropy
        loss = F.cross_entropy(
            shift_logits.view(-1, self.vocab_size),
            shift_labels.view(-1),
            ignore_index=self.pad_token_id,
            reduction='mean'
        )
        
        return loss


def compute_ntp_loss(
    encoder,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    lm_head: nn.Module,
    vocab_size: int,
    pad_token_id: int,
) -> torch.Tensor:
    """
    Standalone function to compute NTP loss.
    
    Args:
        encoder: The context encoder (unwrapped)
        input_ids: [B, L] token IDs
        attention_mask: [B, L] attention mask (1 = real, 0 = pad)
        lm_head: Linear layer for token prediction
        vocab_size: Vocabulary size
        pad_token_id: Padding token ID
    
    Returns:
        NTP loss scalar
    """
    B, L = input_ids.shape
    device = input_ids.device
    
    # Get token embeddings
    token_embeds = encoder.get_input_embeddings()(input_ids)
    
    # Create causal mask [B, 1, L, L]
    causal_mask = torch.tril(torch.ones(L, L, device=device))
    causal_mask = causal_mask.unsqueeze(0).unsqueeze(0).expand(B, 1, L, L)
    
    # Apply padding to causal mask
    padding_mask = attention_mask.unsqueeze(1).unsqueeze(2)  # [B, 1, 1, L]
    causal_mask = causal_mask * padding_mask
    
    # Convert to additive mask
    causal_mask = (1.0 - causal_mask) * -10000.0
    
    # Forward pass
    output = encoder.model(
        inputs_embeds=token_embeds,
        attention_mask=causal_mask,
    )
    hidden_states = output.last_hidden_state
    
    # Get logits and compute loss
    lm_logits = lm_head(hidden_states)
    
    shift_logits = lm_logits[:, :-1, :].contiguous()
    shift_labels = input_ids[:, 1:].contiguous()
    
    loss = F.cross_entropy(
        shift_logits.view(-1, vocab_size),
        shift_labels.view(-1),
        ignore_index=pad_token_id,
        reduction='mean'
    )
    
    return loss