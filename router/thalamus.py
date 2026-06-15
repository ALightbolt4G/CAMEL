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
            "math": [r"\d+", r"\+", r"-", r"\*", r"/", r"equals", r"matrix", r"equation", r"حساب", r"معادلة", r"رياضيات"],
            "code": [r"def\s+", r"fn\s+", r"import\s+", r"class\s+", r"struct\s+", r"return\s+", r"كود", r"برمجة", r"خوارزمية"],
            "history": [r"war", r"century", r"king", r"emperor", r"حرب", r"تاريخ", r"ملك", r"رئيس", r"قديم", r"معركة"],
            "geography": [r"map", r"country", r"river", r"mountain", r"خريطة", r"بلد", r"نهر", r"جبل", r"عاصمة", r"قارة"],
            "rezero": [r"subaru", r"emilia", r"rem", r"echidna", r"arc", r"witch", r"return by death", r"سوبارو", r"إيميليا", r"ريم"]
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
        
        # Calculate the initial hint score (coarse classification)
        for domain, patterns in self.domain_map.items():
            match_count = 0
            for pattern in patterns:
                if re.search(pattern, query_lower):
                    match_count += 1
            
            # A score ceiling is applied, dividing by 3.0 means 3 distinct matches = 1.0 score
            if match_count > 0:
                scores[domain] = min(1.0, match_count / 3.0)
                
        # Action Verb Rule: "Write Python" = code_cell prioritization
        has_write = bool(re.search(r"write|create|build", query_lower))
        has_code = bool(re.search(r"python|code|program|function|script", query_lower))
        
        if has_write and has_code:
            scores["code"] = min(1.0, scores.get("code", 0.0) + 0.4)
            
        return scores
