import torch

class HebbianUpdater:
    """
    Implements Bounded Hebbian Learning for updating Concept Graph weights.
    'Neurons that fire together, wire together.'
    """
    def __init__(self, learning_rate: float = 0.05, decay_rate: float = 0.01):
        self.eta = learning_rate
        self.gamma = decay_rate

    def update_weights(self, W: torch.Tensor, activations: torch.Tensor) -> torch.Tensor:
        """
        Updates the adjacency matrix W based on cell co-activations.
        
        Args:
            W: Current Adjacency Matrix (N x N Tensor)
            activations: Current activation vector (N, Tensor)
            
        Returns:
            W_new: Updated Adjacency Matrix
        """
        # Calculate Co-activation Matrix (Outer product)
        C = torch.outer(activations, activations)
        
        # Bounded Growth and Decay
        # Growth slows down as W approaches 1.0
        growth = self.eta * C * (1.0 - W)
        decay = self.gamma * W
        
        # Update Rule
        W_new = W + growth - decay
        
        # Zeroing Self-connections (Main diagonal must be 0)
        I = torch.eye(W.size(0), device=W.device)
        W_new = W_new * (1.0 - I)
        
        # Final safety clamp
        return torch.clamp(W_new, min=0.0, max=1.0)
