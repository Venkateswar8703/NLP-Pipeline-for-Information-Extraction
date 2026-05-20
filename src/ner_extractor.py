"""Named Entity Recognition (NER) Extraction Module.

Implements transformer-based NER using Hugging Face Transformers and SpaCy
for extracting named entities from preprocessed documents.
"""

import logging
from typing import Dict, List, Optional, Any, Tuple
from collections import defaultdict

import torch
import spacy
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    pipeline as hf_pipeline
)

logger = logging.getLogger(__name__)


class NERExtractor:
    """Extracts named entities using transformer-based models.
    
    Supports both Hugging Face transformer models and SpaCy models
    for named entity recognition, with ensemble capabilities.
    
    Attributes:
        model_name: Name/path of the transformer model.
        device: Computing device (cpu/cuda).
        confidence_threshold: Minimum confidence for entity predictions.
        ner_pipeline: Hugging Face NER pipeline.
        spacy_nlp: SpaCy NLP model for fallback/ensemble.
    """
    
    # Standard entity type mapping for normalization
    ENTITY_TYPE_MAP = {
        'PER': 'PERSON',
        'PERSON': 'PERSON',
        'ORG': 'ORGANIZATION',
        'ORGANIZATION': 'ORGANIZATION',
        'LOC': 'LOCATION',
        'LOCATION': 'LOCATION',
        'GPE': 'LOCATION',
        'DATE': 'DATE',
        'TIME': 'TIME',
        'MONEY': 'MONETARY_VALUE',
        'PERCENT': 'PERCENTAGE',
        'CARDINAL': 'NUMBER',
        'ORDINAL': 'ORDINAL',
        'PRODUCT': 'PRODUCT',
        'EVENT': 'EVENT',
        'FAC': 'FACILITY',
        'NORP': 'GROUP',
        'LAW': 'LAW',
        'LANGUAGE': 'LANGUAGE',
        'WORK_OF_ART': 'WORK_OF_ART',
        'QUANTITY': 'QUANTITY',
    }
    
    def __init__(
        self,
        model_name: str = "dslim/bert-base-NER",
        spacy_model: str = "en_core_web_trf",
        confidence_threshold: float = 0.85,
        device: Optional[str] = None,
        use_ensemble: bool = True,
        aggregation_strategy: str = "simple"
    ):
        """Initialize the NER extractor.
        
        Args:
            model_name: Hugging Face model name or path for NER.
            spacy_model: SpaCy model for ensemble NER.
            confidence_threshold: Minimum confidence threshold.
            device: Device to use ('cpu', 'cuda', 'cuda:0', etc.).
            use_ensemble: Whether to use ensemble of both models.
            aggregation_strategy: Token aggregation strategy ('simple',
                                'first', 'average', 'max').
        """
        self.model_name = model_name
        self.confidence_threshold = confidence_threshold
        self.use_ensemble = use_ensemble
        self.aggregation_strategy = aggregation_strategy
        
        # Determine device
        if device:
            self.device = device
        else:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        device_num = 0 if self.device.startswith("cuda") else -1
        
        # Load transformer model
        logger.info(f"Loading NER model: {model_name} on {self.device}")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForTokenClassification.from_pretrained(model_name)
            self.ner_pipeline = hf_pipeline(
                "ner",
                model=self.model,
                tokenizer=self.tokenizer,
                device=device_num,
                aggregation_strategy=aggregation_strategy
            )
        except Exception as e:
            logger.error(f"Failed to load transformer model: {e}")
            raise
        
        # Load SpaCy model for ensemble
        self.spacy_nlp = None
        if use_ensemble:
            try:
                self.spacy_nlp = spacy.load(spacy_model)
                logger.info(f"Loaded SpaCy model: {spacy_model} for ensemble.")
            except OSError:
                logger.warning(
                    f"SpaCy model '{spacy_model}' not found. "
                    f"Ensemble mode disabled."
                )
                self.use_ensemble = False
        
        logger.info("NERExtractor initialized successfully.")
    
    def _normalize_entity_type(self, entity_type: str) -> str:
        """Normalize entity type to standard format.
        
        Args:
            entity_type: Raw entity type string.
            
        Returns:
            Normalized entity type.
        """
        # Remove B-/I- prefixes from BIO tagging
        clean_type = entity_type.split('-')[-1] if '-' in entity_type else entity_type
        return self.ENTITY_TYPE_MAP.get(clean_type.upper(), clean_type.upper())
    
    def extract_entities_transformer(
        self,
        text: str
    ) -> List[Dict[str, Any]]:
        """Extract entities using the transformer model.
        
        Args:
            text: Input text for entity extraction.
            
        Returns:
            List of entity dictionaries.
        """
        if not text or not text.strip():
            return []
        
        # Handle long texts by chunking
        max_length = self.tokenizer.model_max_length
        chunks = self._chunk_text(text, max_length=max_length - 50)
        
        all_entities = []
        offset = 0
        
        for chunk in chunks:
            try:
                raw_entities = self.ner_pipeline(chunk)
                
                for entity in raw_entities:
                    score = entity.get('score', 0)
                    if score >= self.confidence_threshold:
                        all_entities.append({
                            'text': entity.get('word', '').strip(),
                            'type': self._normalize_entity_type(
                                entity.get('entity_group', entity.get('entity', 'UNKNOWN'))
                            ),
                            'start': entity.get('start', 0) + offset,
                            'end': entity.get('end', 0) + offset,
                            'confidence': round(score, 4),
                            'source': 'transformer'
                        })
            except Exception as e:
                logger.error(f"Error processing chunk: {e}")
                continue
            
            offset += len(chunk)
        
        return all_entities
    
    def extract_entities_spacy(
        self,
        text: str
    ) -> List[Dict[str, Any]]:
        """Extract entities using SpaCy.
        
        Args:
            text: Input text for entity extraction.
            
        Returns:
            List of entity dictionaries.
        """
        if not self.spacy_nlp or not text:
            return []
        
        doc = self.spacy_nlp(text)
        entities = []
        
        for ent in doc.ents:
            entities.append({
                'text': ent.text,
                'type': self._normalize_entity_type(ent.label_),
                'start': ent.start_char,
                'end': ent.end_char,
                'confidence': 0.80,  # SpaCy doesn't provide confidence
                'source': 'spacy'
            })
        
        return entities
    
    def _merge_entities(
        self,
        transformer_entities: List[Dict],
        spacy_entities: List[Dict]
    ) -> List[Dict[str, Any]]:
        """Merge entities from transformer and SpaCy with deduplication.
        
        Uses span overlap detection to merge entities from both models.
        When entities overlap, the higher confidence prediction is kept.
        
        Args:
            transformer_entities: Entities from transformer model.
            spacy_entities: Entities from SpaCy model.
            
        Returns:
            Merged and deduplicated entity list.
        """
        merged = list(transformer_entities)
        
        for spacy_ent in spacy_entities:
            is_duplicate = False
            for trans_ent in transformer_entities:
                # Check for span overlap
                if (spacy_ent['start'] < trans_ent['end'] and
                    spacy_ent['end'] > trans_ent['start']):
                    is_duplicate = True
                    # Keep the one with higher confidence
                    if spacy_ent['confidence'] > trans_ent['confidence']:
                        merged.remove(trans_ent)
                        merged.append(spacy_ent)
                    break
            
            if not is_duplicate:
                merged.append(spacy_ent)
        
        # Sort by position
        merged.sort(key=lambda x: x['start'])
        return merged
    
    def _chunk_text(
        self,
        text: str,
        max_length: int = 450
    ) -> List[str]:
        """Split text into chunks respecting sentence boundaries.
        
        Args:
            text: Text to chunk.
            max_length: Maximum chunk length in characters.
            
        Returns:
            List of text chunks.
        """
        if len(text) <= max_length:
            return [text]
        
        chunks = []
        current_chunk = ""
        
        sentences = text.split('. ')
        
        for sentence in sentences:
            if len(current_chunk) + len(sentence) + 2 <= max_length:
                current_chunk += sentence + '. '
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = sentence + '. '
        
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return chunks
    
    def extract(
        self,
        text: str,
        entity_types: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Extract named entities from text.
        
        Args:
            text: Input text for entity extraction.
            entity_types: Optional filter for specific entity types.
            
        Returns:
            Dictionary with extraction results.
        """
        # Transformer extraction
        transformer_entities = self.extract_entities_transformer(text)
        
        # Ensemble with SpaCy if enabled
        if self.use_ensemble:
            spacy_entities = self.extract_entities_spacy(text)
            entities = self._merge_entities(transformer_entities, spacy_entities)
        else:
            entities = transformer_entities
        
        # Filter by entity types if specified
        if entity_types:
            normalized_types = [
                self._normalize_entity_type(t) for t in entity_types
            ]
            entities = [
                e for e in entities
                if e['type'] in normalized_types
            ]
        
        # Group entities by type
        entities_by_type = defaultdict(list)
        for entity in entities:
            entities_by_type[entity['type']].append(entity)
        
        return {
            'entities': entities,
            'entities_by_type': dict(entities_by_type),
            'total_entities': len(entities),
            'unique_types': list(entities_by_type.keys()),
            'num_types': len(entities_by_type)
        }
