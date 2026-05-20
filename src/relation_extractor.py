"""Relation Extraction Module.

Extracts semantic relationships between identified entities using
transformer-based models for structured information extraction.
"""

import logging
from typing import Dict, List, Optional, Any, Tuple
from itertools import combinations

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    pipeline as hf_pipeline
)

logger = logging.getLogger(__name__)


class RelationExtractor:
    """Extracts relationships between entities in text.
    
    Uses transformer-based models to classify the relationship
    between entity pairs found in documents.
    
    Attributes:
        model_name: Name/path of the relation extraction model.
        device: Computing device.
        confidence_threshold: Minimum confidence for relations.
        max_entity_distance: Maximum token distance between entity pairs.
    """
    
    # Supported relation types
    RELATION_TYPES = [
        'WORKS_FOR', 'LOCATED_IN', 'FOUNDED_BY', 'CEO_OF',
        'SUBSIDIARY_OF', 'PARTNER_OF', 'ACQUIRED_BY', 'INVESTED_IN',
        'BORN_IN', 'EDUCATED_AT', 'MEMBER_OF', 'PART_OF',
        'MANUFACTURED_BY', 'DEVELOPED_BY', 'AUTHORED_BY',
        'PUBLISHED_BY', 'AFFILIATED_WITH', 'REPORTS_TO',
        'HAS_DATE', 'HAS_VALUE', 'NO_RELATION'
    ]
    
    def __init__(
        self,
        model_name: str = "facebook/bart-large-mnli",
        confidence_threshold: float = 0.70,
        max_entity_distance: int = 200,
        device: Optional[str] = None,
        max_entity_pairs: int = 500
    ):
        """Initialize the relation extractor.
        
        Args:
            model_name: Hugging Face model for zero-shot classification.
            confidence_threshold: Minimum confidence for relation predictions.
            max_entity_distance: Max character distance between entity pairs.
            device: Computing device.
            max_entity_pairs: Maximum number of entity pairs to evaluate.
        """
        self.model_name = model_name
        self.confidence_threshold = confidence_threshold
        self.max_entity_distance = max_entity_distance
        self.max_entity_pairs = max_entity_pairs
        
        if device:
            self.device = device
        else:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        device_num = 0 if self.device.startswith("cuda") else -1
        
        logger.info(
            f"Loading relation extraction model: {model_name} on {self.device}"
        )
        
        try:
            self.classifier = hf_pipeline(
                "zero-shot-classification",
                model=model_name,
                device=device_num
            )
        except Exception as e:
            logger.error(f"Failed to load relation model: {e}")
            raise
        
        logger.info("RelationExtractor initialized successfully.")
    
    def _generate_entity_pairs(
        self,
        entities: List[Dict[str, Any]]
    ) -> List[Tuple[Dict, Dict]]:
        """Generate candidate entity pairs for relation extraction.
        
        Filters pairs based on distance and entity type compatibility.
        
        Args:
            entities: List of entity dictionaries.
            
        Returns:
            List of entity pair tuples.
        """
        pairs = []
        
        for e1, e2 in combinations(entities, 2):
            # Skip pairs of same entity
            if e1['text'].lower() == e2['text'].lower():
                continue
            
            # Check distance constraint
            distance = abs(e1['start'] - e2['start'])
            if distance > self.max_entity_distance:
                continue
            
            pairs.append((e1, e2))
        
        # Limit number of pairs
        if len(pairs) > self.max_entity_pairs:
            logger.warning(
                f"Too many entity pairs ({len(pairs)}). "
                f"Limiting to {self.max_entity_pairs}."
            )
            pairs = pairs[:self.max_entity_pairs]
        
        return pairs
    
    def _get_context_window(
        self,
        text: str,
        entity1: Dict,
        entity2: Dict,
        window_size: int = 100
    ) -> str:
        """Extract the context window containing both entities.
        
        Args:
            text: Full document text.
            entity1: First entity dictionary.
            entity2: Second entity dictionary.
            window_size: Extra characters to include around entities.
            
        Returns:
            Context string containing both entities.
        """
        start = max(0, min(entity1['start'], entity2['start']) - window_size)
        end = min(
            len(text),
            max(entity1['end'], entity2['end']) + window_size
        )
        return text[start:end]
    
    def _classify_relation(
        self,
        text: str,
        entity1: Dict,
        entity2: Dict
    ) -> Dict[str, Any]:
        """Classify the relation between two entities.
        
        Args:
            text: Context text containing both entities.
            entity1: First entity.
            entity2: Second entity.
            
        Returns:
            Relation dictionary with type and confidence.
        """
        candidate_labels = [
            label.lower().replace('_', ' ')
            for label in self.RELATION_TYPES
            if label != 'NO_RELATION'
        ]
        
        try:
            result = self.classifier(
                text,
                candidate_labels=candidate_labels,
                hypothesis_template=(
                    f"{entity1['text']} {{}} {entity2['text']}."
                ),
                multi_label=False
            )
            
            top_label = result['labels'][0]
            top_score = result['scores'][0]
            
            if top_score >= self.confidence_threshold:
                return {
                    'subject': entity1,
                    'object': entity2,
                    'relation': top_label.upper().replace(' ', '_'),
                    'confidence': round(top_score, 4),
                    'context': text[:200]
                }
        except Exception as e:
            logger.error(
                f"Error classifying relation between "
                f"'{entity1['text']}' and '{entity2['text']}': {e}"
            )
        
        return None
    
    def extract_relations(
        self,
        text: str,
        entities: List[Dict[str, Any]],
        relation_types: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Extract relations between entities in text.
        
        Args:
            text: Source document text.
            entities: List of extracted entities.
            relation_types: Optional filter for specific relation types.
            
        Returns:
            Dictionary with extraction results.
        """
        if not entities or len(entities) < 2:
            return {
                'relations': [],
                'total_relations': 0,
                'entity_pairs_evaluated': 0
            }
        
        # Generate entity pairs
        pairs = self._generate_entity_pairs(entities)
        logger.info(f"Evaluating {len(pairs)} entity pairs for relations.")
        
        relations = []
        
        for entity1, entity2 in pairs:
            # Get context window
            context = self._get_context_window(text, entity1, entity2)
            
            # Classify relation
            relation = self._classify_relation(context, entity1, entity2)
            
            if relation:
                # Apply relation type filter
                if relation_types:
                    normalized_types = [
                        rt.upper().replace(' ', '_')
                        for rt in relation_types
                    ]
                    if relation['relation'] not in normalized_types:
                        continue
                
                relations.append(relation)
        
        logger.info(f"Extracted {len(relations)} relations.")
        
        return {
            'relations': relations,
            'total_relations': len(relations),
            'entity_pairs_evaluated': len(pairs),
            'unique_relation_types': list(set(
                r['relation'] for r in relations
            ))
        }
