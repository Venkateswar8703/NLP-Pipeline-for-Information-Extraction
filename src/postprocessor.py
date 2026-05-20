"""Post-processing Module.

Handles deduplication, normalization, confidence filtering, and
structured output formatting for extracted entities and relations.
"""

import logging
from typing import Dict, List, Optional, Any
from collections import defaultdict
from difflib import SequenceMatcher

logger = logging.getLogger(__name__)


class PostProcessor:
    """Post-processes extracted entities and relations.
    
    Performs deduplication, normalization, confidence filtering,
    and formats output for downstream consumption.
    
    Attributes:
        dedup_threshold: Similarity threshold for deduplication.
        min_entity_length: Minimum entity text length.
    """
    
    def __init__(
        self,
        dedup_threshold: float = 0.90,
        min_entity_length: int = 2,
        normalize_case: bool = True
    ):
        """Initialize the post-processor.
        
        Args:
            dedup_threshold: Similarity threshold for entity deduplication.
            min_entity_length: Minimum character length for valid entities.
            normalize_case: Whether to normalize entity text case.
        """
        self.dedup_threshold = dedup_threshold
        self.min_entity_length = min_entity_length
        self.normalize_case = normalize_case
        
        logger.info("PostProcessor initialized.")
    
    def _text_similarity(self, text1: str, text2: str) -> float:
        """Calculate text similarity using SequenceMatcher.
        
        Args:
            text1: First text string.
            text2: Second text string.
            
        Returns:
            Similarity score between 0 and 1.
        """
        return SequenceMatcher(
            None,
            text1.lower().strip(),
            text2.lower().strip()
        ).ratio()
    
    def deduplicate_entities(
        self,
        entities: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Remove duplicate entities based on text similarity.
        
        When duplicates are found, the entity with the highest
        confidence score is retained.
        
        Args:
            entities: List of entity dictionaries.
            
        Returns:
            Deduplicated list of entities.
        """
        if not entities:
            return []
        
        unique_entities = []
        seen_texts = []
        
        # Sort by confidence (descending) so we keep highest confidence
        sorted_entities = sorted(
            entities,
            key=lambda x: x.get('confidence', 0),
            reverse=True
        )
        
        for entity in sorted_entities:
            entity_text = entity.get('text', '').strip()
            
            if len(entity_text) < self.min_entity_length:
                continue
            
            is_duplicate = False
            for seen in seen_texts:
                if self._text_similarity(entity_text, seen) >= self.dedup_threshold:
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                unique_entities.append(entity)
                seen_texts.append(entity_text)
        
        logger.debug(
            f"Deduplication: {len(entities)} -> {len(unique_entities)} entities."
        )
        
        return sorted(unique_entities, key=lambda x: x.get('start', 0))
    
    def normalize_entities(
        self,
        entities: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Normalize entity text and types.
        
        Args:
            entities: List of entity dictionaries.
            
        Returns:
            Normalized list of entities.
        """
        normalized = []
        
        for entity in entities:
            norm_entity = dict(entity)
            
            # Clean entity text
            text = norm_entity.get('text', '').strip()
            
            # Remove leading/trailing punctuation
            text = text.strip('.,;:!?()[]{}\"\'')
            
            if len(text) < self.min_entity_length:
                continue
            
            # Normalize case for specific entity types
            if self.normalize_case:
                entity_type = norm_entity.get('type', '')
                if entity_type in ('PERSON', 'ORGANIZATION', 'LOCATION'):
                    text = text.title()
            
            norm_entity['text'] = text
            normalized.append(norm_entity)
        
        return normalized
    
    def format_structured_output(
        self,
        result: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Format extraction results into structured output.
        
        Args:
            result: Raw extraction result dictionary.
            
        Returns:
            Formatted structured output.
        """
        entities = result.get('entities', {}).get('entities', [])
        relations = result.get('relations', {}).get('relations', [])
        
        # Build knowledge graph representation
        knowledge_graph = {
            'nodes': [],
            'edges': []
        }
        
        # Add entity nodes
        entity_ids = {}
        for i, entity in enumerate(entities):
            node_id = f"entity_{i}"
            entity_ids[f"{entity['text']}_{entity['type']}"] = node_id
            knowledge_graph['nodes'].append({
                'id': node_id,
                'label': entity['text'],
                'type': entity['type'],
                'confidence': entity.get('confidence', 0)
            })
        
        # Add relation edges
        for i, relation in enumerate(relations):
            subject_key = (
                f"{relation['subject']['text']}_{relation['subject']['type']}"
            )
            object_key = (
                f"{relation['object']['text']}_{relation['object']['type']}"
            )
            
            knowledge_graph['edges'].append({
                'id': f"relation_{i}",
                'source': entity_ids.get(subject_key, 'unknown'),
                'target': entity_ids.get(object_key, 'unknown'),
                'relation': relation['relation'],
                'confidence': relation.get('confidence', 0)
            })
        
        result['knowledge_graph'] = knowledge_graph
        
        # Add summary statistics
        result['summary'] = {
            'total_entities': len(entities),
            'total_relations': len(relations),
            'entity_types': list(set(e['type'] for e in entities)),
            'relation_types': list(set(r['relation'] for r in relations)),
            'avg_entity_confidence': round(
                sum(e.get('confidence', 0) for e in entities) / max(len(entities), 1),
                4
            ),
            'avg_relation_confidence': round(
                sum(r.get('confidence', 0) for r in relations) / max(len(relations), 1),
                4
            )
        }
        
        return result
    
    def process(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Run full post-processing pipeline.
        
        Args:
            result: Raw extraction result from the pipeline.
            
        Returns:
            Post-processed result dictionary.
        """
        # Deduplicate entities
        if 'entities' in result and 'entities' in result['entities']:
            entities = result['entities']['entities']
            
            # Normalize
            entities = self.normalize_entities(entities)
            
            # Deduplicate
            entities = self.deduplicate_entities(entities)
            
            # Update result
            result['entities']['entities'] = entities
            result['entities']['total_entities'] = len(entities)
            
            # Regroup by type
            entities_by_type = defaultdict(list)
            for entity in entities:
                entities_by_type[entity['type']].append(entity)
            result['entities']['entities_by_type'] = dict(entities_by_type)
        
        # Format structured output
        result = self.format_structured_output(result)
        
        return result
