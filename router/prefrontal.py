import torch
import torch.nn.functional as F
from typing import Dict

class PrefrontalCortex:
    """
    Level 2: Prefrontal Cortex (PFC)
    Performs deep semantic analysis using shared embeddings and domain centroids.
    Employs a gating mechanism to save computation if Thalamus is highly confident.
    """
    def __init__(self, shared_embedding_layer: torch.nn.Embedding, domain_centroids: Dict[str, torch.Tensor]):
        """
        Args:
            shared_embedding_layer: The shared token embedding layer across cells.
            domain_centroids: A dictionary mapping domain names to their centroid vectors (1D Tensors).
        """
        self.embedding = shared_embedding_layer
        self.centroids = domain_centroids
        
    def evaluate(self, tokenized_query: torch.Tensor, thalamus_hint: Dict[str, float]) -> Dict[str, float]:
        """
        Evaluate the query against the domain centroids and combine with Thalamus hints.
        
        Args:
            tokenized_query: Tensor of shape (1, seq_len) containing input token IDs.
            thalamus_hint: Dictionary of initial scores from Thalamus.
            
        Returns:
            final_scores: Dictionary of combined scores.
        """
        # Gating Mechanism: If Thalamus is highly confident, save computation (Branchless-like efficiency)
        max_hint = max(thalamus_hint.values()) if thalamus_hint else 0.0
        if max_hint > 0.8:
            return thalamus_hint 

        # Calculate the Query Vector using Mean Pooling
        with torch.no_grad():
            # embedding shape: (1, seq_len, embed_dim)
            # mean pooling shape: (1, embed_dim)
            # squeeze shape: (embed_dim,)
            query_emb = self.embedding(tokenized_query).mean(dim=1).squeeze(0) 
            
        final_scores = {}
        for domain, hint_score in thalamus_hint.items():
            if domain in self.centroids:
                centroid = self.centroids[domain]
                # Calculate Cosine Similarity
                sim = F.cosine_similarity(query_emb, centroid, dim=0).item()
                # Ensure the similarity doesn't drop the score to negative
                sim = max(0.0, sim)
                
                # Merge Signals (40% Thalamus, 60% PFC)
                final_scores[domain] = (hint_score * 0.4) + (sim * 0.6)
            else:
                final_scores[domain] = hint_score
                
        return final_scores
