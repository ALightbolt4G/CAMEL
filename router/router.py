from typing import Dict, List, Tuple
from .thalamus import Thalamus

class CamelRouter:
    """
    Biological Router: Routes context and data between CAMEL Cells.
    Currently utilizes the Thalamus (Level 1) for rapid coarse-grained routing.
    """
    def __init__(self):
        self.thalamus = Thalamus()
        self.activation_threshold = 0.2

    def route(self, query: str) -> Tuple[List[str], Dict[str, float]]:
        """
        Routes the query to determine active cells.
        
        Args:
            query: The input text.
            
        Returns:
            Tuple of (active_cells_list, raw_scores_dict)
        """
        raw_scores = self.thalamus.route(query)
        
        active_cells = []
        cell_scores = {}
        
        for domain, score in raw_scores.items():
            cell_name = f"{domain}_cell"
            # Normalize missing scores to baseline 0.05
            final_score = score if score > 0 else 0.05
            cell_scores[cell_name] = final_score
            
            # Use threshold to activate
            if final_score >= self.activation_threshold:
                active_cells.append(cell_name)
                
        return active_cells, cell_scores
