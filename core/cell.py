import torch
import torch.nn as nn

class CamelCell(nn.Module):
    """
    CAMEL Cell: A Tiny Transformer specialized in a single domain.
    Outputs an embedding vector representing the semantic meaning, rather than raw tokens.
    """
    def __init__(self, vocab_size=32000, d_model=128, n_heads=4, num_layers=4, dim_feedforward=512, max_seq_len=512, dropout=0.1):
        super(CamelCell, self).__init__()
        
        self.d_model = d_model
        
        # Token and Positional Embeddings
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.positional_embedding = nn.Embedding(max_seq_len, d_model)
        
        # Transformer Encoder (Tiny Transformer)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Output projection to standardized vector space
        # (This implements the MSLM feature: "المخرج: embedding vector (مش tokens مباشرة)")
        self.output_projection = nn.Linear(d_model, d_model)
        
        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, input_ids, padding_mask=None):
        """
        Forward pass for the cell.
        
        Args:
            input_ids: Tensor of shape (batch_size, seq_len)
            padding_mask: Boolean Tensor of shape (batch_size, seq_len) - True for pad tokens
            
        Returns:
            output_vector: Tensor of shape (batch_size, d_model)
        """
        batch_size, seq_len = input_ids.size()
        
        # Generate positional ids
        positions = torch.arange(0, seq_len, dtype=torch.long, device=input_ids.device)
        positions = positions.unsqueeze(0).expand(batch_size, seq_len)
        
        # Apply Embeddings
        x = self.token_embedding(input_ids) + self.positional_embedding(positions)
        
        # Pass through Tiny Transformer
        x = self.transformer(x, src_key_padding_mask=padding_mask)
        
        # In MSLM, cells output a context vector. We use mean pooling over the sequence.
        if padding_mask is not None:
            # Mask out padding tokens before pooling
            mask_expanded = (~padding_mask).unsqueeze(-1).float()
            sum_embeddings = torch.sum(x * mask_expanded, dim=1)
            sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
            pooled_output = sum_embeddings / sum_mask
        else:
            pooled_output = torch.mean(x, dim=1)
            
        # Final projection to domain embedding
        output_vector = self.output_projection(pooled_output)
        
        return output_vector

    def count_parameters(self):
        """Returns the number of trainable parameters in millions"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad) / 1e6
