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


def compute_ntp_loss_code_only(
    encoder,
    text_input_ids: torch.Tensor,
    code_input_ids: torch.Tensor,
    text_attention_mask: torch.Tensor,
    code_attention_mask: torch.Tensor,
    lm_head: nn.Module,
    vocab_size: int,
    pad_token_id: int,
) -> torch.Tensor:
    """
    Compute NTP loss on concatenated [text, code] but only supervise code tokens.

    The input is formed as [text_ids | code_ids] with a causal mask.
    Cross-entropy is computed only on code positions (text labels are ignored).

    Args:
        encoder: The context encoder (unwrapped)
        text_input_ids: [B, L_t] text token IDs
        code_input_ids: [B, L_c] code token IDs
        text_attention_mask: [B, L_t] attention mask for text
        code_attention_mask: [B, L_c] attention mask for code
        lm_head: Linear layer for token prediction
        vocab_size: Vocabulary size
        pad_token_id: Padding token ID

    Returns:
        NTP loss scalar (CE on code tokens only)
    """
    B = text_input_ids.shape[0]
    L_t = text_input_ids.shape[1]
    L_c = code_input_ids.shape[1]
    device = text_input_ids.device

    # Concatenate text and code
    concat_ids = torch.cat([text_input_ids, code_input_ids], dim=1)  # [B, L_t+L_c]
    concat_mask = torch.cat([text_attention_mask, code_attention_mask], dim=1)  # [B, L_t+L_c]
    L = L_t + L_c

    # Get token embeddings
    token_embeds = encoder.get_input_embeddings()(concat_ids)

    # Create causal mask [B, 1, L, L]
    causal_mask = torch.tril(torch.ones(L, L, device=device))
    causal_mask = causal_mask.unsqueeze(0).unsqueeze(0).expand(B, 1, L, L).clone()

    # Apply padding
    padding_mask = concat_mask.unsqueeze(1).unsqueeze(2).float()  # [B, 1, 1, L]
    causal_mask = causal_mask * padding_mask

    # Convert to additive mask
    causal_mask = (1.0 - causal_mask) * -10000.0

    # Forward pass
    output = encoder.model(
        inputs_embeds=token_embeds,
        attention_mask=causal_mask,
    )
    hidden_states = output.last_hidden_state

    # Get logits
    lm_logits = lm_head(hidden_states)

    # Shift for causal prediction
    shift_logits = lm_logits[:, :-1, :].contiguous()  # [B, L-1, V]
    shift_labels = concat_ids[:, 1:].contiguous()       # [B, L-1]

    # Build label mask: ignore text positions, only supervise code positions.
    # After shifting, position i predicts token i+1.
    # Code tokens occupy positions L_t .. L_t+L_c-1 in concat_ids.
    # So shifted labels for code start at index L_t-1 (predicting token at L_t).
    ignore_mask = torch.full_like(shift_labels, pad_token_id)
    ignore_mask[:, L_t - 1:] = shift_labels[:, L_t - 1:]
    # Also mask out padding within code region
    shift_code_mask = concat_mask[:, 1:]  # [B, L-1]
    ignore_mask = torch.where(shift_code_mask == 1, ignore_mask, torch.tensor(pad_token_id, device=device))

    loss = F.cross_entropy(
        shift_logits.view(-1, vocab_size),
        ignore_mask.view(-1),
        ignore_index=pad_token_id,
        reduction='mean'
    )

    return loss