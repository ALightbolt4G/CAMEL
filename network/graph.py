import torch

class ConceptGraph:
    """
    Concept Graph for the MSLM Network.
    Manages connections between domains using an Adjacency Matrix and implements Spreading Activation.
    """
    def __init__(self, domains: list, alpha: float = 0.2):
        self.domains = domains
        self.domain_to_idx = {d: i for i, d in enumerate(domains)}
        self.n = len(domains)
        self.alpha = alpha
        
        # N x N Adjacency Matrix (W)
        self.W = torch.zeros((self.n, self.n))
        
        # Initialize Heuristic Connections
        self._initialize_heuristics()
        
    def _initialize_heuristics(self):
        """
        Set initial connections based on cognitive proximity.
        These will be updated later via Hebbian Learning.
        """
        def connect(d1, d2, weight):
            if d1 in self.domain_to_idx and d2 in self.domain_to_idx:
                i, j = self.domain_to_idx[d1], self.domain_to_idx[d2]
                self.W[i, j] = weight
                self.W[j, i] = weight # Undirected graph for now
                
        # Heuristic Connections
        connect("math", "code", 0.7)
        connect("history", "geography", 0.6)
        # Math & Geography? No obvious strong connection initially -> 0.0

    def spread_activation(self, initial_activations: torch.Tensor, hops: int = 1) -> torch.Tensor:
        """
        Spreads the activation through the graph using matrix multiplication.
        
        Args:
            initial_activations: Tensor of shape (N,) representing current cell activations.
            hops: Number of spreading hops. 1 hop is usually sufficient and extremely fast O(1).
            
        Returns:
            new_activations: Tensor of shape (N,) clamped between 0 and 1.
        """
        current_activations = initial_activations
        
        for _ in range(hops):
            # Matrix multiplication: S = W * A_t
            spread_signal = torch.matmul(self.W, current_activations)
            
            # Combine: A_{t+1} = A_t + (alpha * S)
            current_activations = current_activations + (self.alpha * spread_signal)
            
            # Clamp between 0 and 1 to prevent exploding activations
            current_activations = torch.clamp(current_activations, min=0.0, max=1.0)
            
        return current_activations
