import torch
import torch.nn as nn

class BaselineTransformer(nn.Module):
    """
    The Dense Baseline Model (Monolithic Transformer).
    Designed to have roughly the same total parameter count as the combined CAMEL network (~20M params).
    Uses Weight Tying for the LM Head to mirror CAMEL's setup for a fair Apple-to-Apple comparison.
    """
    def __init__(self, vocab_size=32000, d_model=256, n_heads=8, num_layers=8, dim_feedforward=1024, max_seq_len=512, dropout=0.1):
        super(BaselineTransformer, self).__init__()
        
        self.d_model = d_model
        
        # Token and Positional Embeddings
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.positional_embedding = nn.Embedding(max_seq_len, d_model)
        
        # Dense Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Weight Tying: The LM Head uses the transposed token embedding weights
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.token_embedding.weight
        
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, input_ids, padding_mask=None):
        batch_size, seq_len = input_ids.size()
        
        # Generate positional ids
        positions = torch.arange(0, seq_len, dtype=torch.long, device=input_ids.device)
        positions = positions.unsqueeze(0).expand(batch_size, seq_len)
        
        # Embeddings
        x = self.token_embedding(input_ids) + self.positional_embedding(positions)
        
        # Dense computation: All parameters process every token
        x = self.transformer(x, src_key_padding_mask=padding_mask)
        
        # Output Logits
        logits = self.lm_head(x)
        return logits

    def count_parameters(self):
        """Returns the number of trainable parameters in millions"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad) / 1e6
