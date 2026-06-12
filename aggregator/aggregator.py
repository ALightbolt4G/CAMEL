import torch
import torch.nn as nn

class CamelAggregator(nn.Module):
    """
    Combines the embeddings from active CAMEL cells using a weighted sum based on activation scores.
    Employs Weight Tying with the shared embedding layer to produce logits without extra parameters.
    """
    def __init__(self, shared_embedding_weight: torch.Tensor):
        super().__init__()
        self.vocab_size, self.embed_dim = shared_embedding_weight.shape
        
        # Weight Tying for maximum VRAM efficiency
        # Linear layer without bias for the LM Head
        self.lm_head = nn.Linear(self.embed_dim, self.vocab_size, bias=False)
        self.lm_head.weight = shared_embedding_weight

    def forward(self, cell_outputs: dict, activation_scores: dict) -> torch.Tensor:
        """
        Args:
            cell_outputs: dict mapping domain -> Tensor of shape (batch, seq_len, embed_dim)
            activation_scores: dict mapping domain -> float score (0.0 to 1.0)
            
        Returns:
            logits: Tensor of shape (batch, seq_len, vocab_size)
        """
        if not cell_outputs:
            raise ValueError("No active cells provided to the aggregator.")
            
        # Initialize an empty tensor with the same shape and device as the first output
        first_domain = list(cell_outputs.keys())[0]
        aggregated_embeds = torch.zeros_like(cell_outputs[first_domain])
        
        # Weighted sum of active cell embeddings
        for domain, output in cell_outputs.items():
            score = activation_scores.get(domain, 0.0)
            if score > 0.0:
                aggregated_embeds += output * score
                
        # Convert continuous vectors to Logits over the vocabulary
        logits = self.lm_head(aggregated_embeds)
        return logits
