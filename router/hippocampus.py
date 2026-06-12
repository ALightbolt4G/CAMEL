from typing import Dict

class Hippocampus:
    """
    Level 3: Hippocampus
    Provides Context Memory using an Exponential Decay (Low-pass filter) state.
    Includes a 'Surprise / Context Switch' detection to flush memory on abrupt topic changes.
    """
    def __init__(self, decay_factor: float = 0.7, boost_weight: float = 0.3, switch_threshold: float = 0.7):
        self.decay = decay_factor
        self.boost_weight = boost_weight
        self.switch_threshold = switch_threshold
        # State vector for context (O(1) memory)
        self.memory_state: Dict[str, float] = {}

    def apply_context(self, current_scores: Dict[str, float]) -> Dict[str, float]:
        final_activation = {}
        
        # 1. Context Switch (Surprise) Detection
        # Identify the dominant domain in the new input and in our memory
        max_current_domain = max(current_scores, key=current_scores.get) if current_scores else None
        max_current_score = current_scores.get(max_current_domain, 0.0)
        
        if self.memory_state:
            max_memory_domain = max(self.memory_state, key=self.memory_state.get)
            
            # If the user suddenly asks a very clear question (score > threshold) 
            # in a completely different domain, we trigger an "Attention Reset"
            if max_current_domain != max_memory_domain and max_current_score >= self.switch_threshold:
                # Flush memory to prevent old context from dragging the new topic down
                for k in self.memory_state.keys():
                    self.memory_state[k] = 0.0
                    
        # 2. Apply Memory Boost & Update State
        for domain, pfc_score in current_scores.items():
            if domain not in self.memory_state:
                self.memory_state[domain] = 0.0
                
            # Calculate decayed memory influence
            decayed_memory = self.memory_state[domain] * self.decay
            
            # Boost the current score
            boosted_score = pfc_score + (decayed_memory * self.boost_weight)
            final_activation[domain] = min(1.0, boosted_score)
            
            # Update memory state using Exponential Moving Average
            self.memory_state[domain] = (self.decay * self.memory_state[domain]) + ((1.0 - self.decay) * final_activation[domain])
            
        return final_activation
