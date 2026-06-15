import torch
import torch.nn.functional as F

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None

class PrefrontalCortex:
    """
    Level 2: Prefrontal Cortex (PFC)
    Performs deep semantic analysis using sentence embeddings (all-MiniLM-L6-v2) 
    and cosine similarity against each cell's unique signature.
    """
    def __init__(self):
        if SentenceTransformer is None:
            raise ImportError("Please install sentence-transformers: pip install sentence-transformers")
        
        # Load lightweight and fast embedding model
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        
        # Cell Signatures (Zero-shot configuration)
        self.signatures = {
            "code": "programming, Python, algorithms, functions",
            "math": "mathematics, calculus, equations, algebra",
            "history": "history, wars, events, civilizations",
            "rezero": "anime, light novel, subaru, emilia, witches, fantasy story, arc, return by death, isekai"
        }
        
        # Pre-compute signature embeddings
        self.signature_embeddings = {}
        for domain, signature in self.signatures.items():
            emb = self.model.encode(signature, convert_to_tensor=True)
            # Ensure it's a 1D tensor
            if emb.dim() > 1:
                emb = emb.squeeze()
            self.signature_embeddings[domain] = emb
            
    def evaluate(self, query: str) -> dict:
        """
        Evaluate the query against the domain signatures.
        
        Args:
            query: The user input string.
            
        Returns:
            scores: Dictionary mapping domain to a cosine similarity score (normalized 0 to 1).
        """
        query_emb = self.model.encode(query, convert_to_tensor=True)
        if query_emb.dim() > 1:
            query_emb = query_emb.squeeze()
            
        scores = {}
        for domain, sig_emb in self.signature_embeddings.items():
            sim = F.cosine_similarity(query_emb, sig_emb, dim=0).item()
            # Cosine similarity is between -1 and 1. 
            # We can clamp to 0-1 or map it to 0-1, but simply clamping negative similarities to 0 is common
            sim = max(0.0, sim)
            scores[domain] = sim
            
        return scores
