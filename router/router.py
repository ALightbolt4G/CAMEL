from typing import Dict, List, Tuple
from .thalamus import Thalamus
from .prefrontal import PrefrontalCortex

class CamelRouter:
    """
    Biological Router: Routes context and data between CAMEL Cells.
    Integrates Thalamus (Fast Keyword Matching) and Prefrontal Cortex (Semantic Embedding Matching).
    """
    def __init__(self):
        self.thalamus = Thalamus()
        self.prefrontal = PrefrontalCortex()
        self.activation_threshold = 0.2

    def route(self, query: str) -> Tuple[List[str], Dict[str, float]]:
        """
        Routes the query to determine active cells.
        
        Args:
            query: The input text.
            
        Returns:
            Tuple of (active_cells_list, combined_scores_dict)
        """
        thalamus_scores = self.thalamus.route(query)
        prefrontal_scores = self.prefrontal.evaluate(query)
        
        active_cells = []
        cell_scores = {}
        
        for domain in ["math", "code", "history", "rezero"]:
            t_score = thalamus_scores.get(domain, 0.0)
            p_score = prefrontal_scores.get(domain, 0.0)
            
            # Combine scores: (thalamus × 0.3) + (prefrontal × 0.7)
            final_score = (t_score * 0.3) + (p_score * 0.7)
            
            cell_name = f"{domain}_cell"
            
            # Ensure baseline
            final_score = final_score if final_score > 0 else 0.05
            cell_scores[cell_name] = final_score
            
            if final_score >= self.activation_threshold:
                active_cells.append(cell_name)
                
        return active_cells, cell_scores
