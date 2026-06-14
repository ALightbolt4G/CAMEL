import re
from typing import Dict, List

class Thalamus:
    """
    Level 1: Thalamus
    Provides rapid, coarse-grained routing using a Gene Map (Regex-based Keyword Matrix).
    Zero VRAM consumption, executes in O(N) time relative to query length.
    """
    def __init__(self):
        # Gene Map: Keywords and patterns representing each domain (English and Arabic)
        self.domain_map: Dict[str, List[str]] = {
            "math": [r"\d+", r"\+", r"-", r"\*", r"/", r"equals", r"matrix", r"equation", r"math", r"calculate", r"probability", r"induction", r"derivative", r"squared", r"differential", r"calculus"],
            "code": [r"def\s+", r"fn\s+", r"import\s+", r"class\s+", r"struct\s+", r"return\s+", r"python", r"program", r"code", r"recursion", r"programming", r"function", r"object oriented", r"decorator"],
            "history": [r"war", r"century", r"king", r"emperor", r"history", r"wwi", r"wwii", r"european", r"economic", r"world war", r"alliance", r"treaty", r"versailles", r"pacific"],
            "geography": [r"map", r"country", r"river", r"mountain", r"continent", r"geography", r"capital"]
        }
        
    def route(self, query: str) -> Dict[str, float]:
        """
        Fast scan of the query against the Gene Map to produce an initial domain hint.
        
        Args:
            query: The user input string.
            
        Returns:
            scores: Dict mapping domain to an initial score between 0.0 and 1.0.
        """
        query_lower = query.lower()
        scores = {domain: 0.0 for domain in self.domain_map.keys()}
        
        # Coarse and extremely fast classification
        for domain, patterns in self.domain_map.items():
            match_count = 0
            for pattern in patterns:
                if re.search(pattern, query_lower):
                    match_count += 1
            
            # Calculate the initial hint score (coarse classification)
            # A score ceiling is applied, dividing by 3.0 means 3 distinct matches = 1.0 score
            if match_count > 0:
                scores[domain] = min(1.0, match_count / 3.0)
                
        return scores
