"""Main NLP Pipeline Orchestrator.

Orchestrates the end-to-end NLP pipeline for structured information extraction,
coordinating preprocessing, NER, relation extraction, and post-processing.
"""

import json
import logging
import time
from typing import Dict, List, Optional, Any
from pathlib import Path
from datetime import datetime

from src.preprocessor import DocumentPreprocessor
from src.ner_extractor import NERExtractor
from src.relation_extractor import RelationExtractor
from src.postprocessor import PostProcessor
from src.utils import setup_logging, load_config, Timer

logger = logging.getLogger(__name__)


class NLPPipeline:
    """End-to-end NLP pipeline for structured information extraction.
    
    Orchestrates document preprocessing, named entity recognition,
    relation extraction, and post-processing into a unified pipeline.
    
    Attributes:
        config: Pipeline configuration dictionary.
        preprocessor: Document preprocessor instance.
        ner_extractor: NER extractor instance.
        relation_extractor: Relation extractor instance.
        postprocessor: Post-processor instance.
    """
    
    def __init__(
        self,
        config_path: Optional[str] = None,
        config: Optional[Dict] = None
    ):
        """Initialize the NLP pipeline.
        
        Args:
            config_path: Path to YAML configuration file.
            config: Configuration dictionary (overrides config_path).
        """
        # Load configuration
        if config:
            self.config = config
        elif config_path:
            self.config = load_config(config_path)
        else:
            self.config = self._default_config()
        
        # Setup logging
        log_level = self.config.get('logging', {}).get('level', 'INFO')
        setup_logging(level=log_level)
        
        logger.info("Initializing NLP Pipeline...")
        
        # Initialize components
        self._init_components()
        
        # Track processing statistics
        self.stats = {
            'documents_processed': 0,
            'total_entities_extracted': 0,
            'total_relations_extracted': 0,
            'total_processing_time': 0.0,
            'errors': []
        }
        
        logger.info("NLP Pipeline initialized successfully.")
    
    @staticmethod
    def _default_config() -> Dict:
        """Return default pipeline configuration."""
        return {
            'preprocessor': {
                'spacy_model': 'en_core_web_sm',
                'max_length': 1_000_000
            },
            'ner': {
                'model_name': 'dslim/bert-base-NER',
                'spacy_model': 'en_core_web_sm',
                'confidence_threshold': 0.85,
                'use_ensemble': True,
                'aggregation_strategy': 'simple'
            },
            'relation_extraction': {
                'model_name': 'facebook/bart-large-mnli',
                'confidence_threshold': 0.70,
                'max_entity_distance': 200
            },
            'postprocessor': {
                'dedup_threshold': 0.90,
                'min_entity_length': 2
            },
            'logging': {
                'level': 'INFO'
            }
        }
    
    def _init_components(self):
        """Initialize pipeline components from configuration."""
        # Preprocessor
        prep_config = self.config.get('preprocessor', {})
        self.preprocessor = DocumentPreprocessor(
            spacy_model=prep_config.get('spacy_model', 'en_core_web_sm'),
            max_length=prep_config.get('max_length', 1_000_000)
        )
        
        # NER Extractor
        ner_config = self.config.get('ner', {})
        self.ner_extractor = NERExtractor(
            model_name=ner_config.get('model_name', 'dslim/bert-base-NER'),
            spacy_model=ner_config.get('spacy_model', 'en_core_web_sm'),
            confidence_threshold=ner_config.get('confidence_threshold', 0.85),
            use_ensemble=ner_config.get('use_ensemble', True),
            aggregation_strategy=ner_config.get(
                'aggregation_strategy', 'simple'
            )
        )
        
        # Relation Extractor
        rel_config = self.config.get('relation_extraction', {})
        self.relation_extractor = RelationExtractor(
            model_name=rel_config.get('model_name', 'facebook/bart-large-mnli'),
            confidence_threshold=rel_config.get('confidence_threshold', 0.70),
            max_entity_distance=rel_config.get('max_entity_distance', 200)
        )
        
        # Post-processor
        post_config = self.config.get('postprocessor', {})
        self.postprocessor = PostProcessor(
            dedup_threshold=post_config.get('dedup_threshold', 0.90),
            min_entity_length=post_config.get('min_entity_length', 2)
        )
    
    def process_document(
        self,
        text: str,
        doc_id: Optional[str] = None,
        metadata: Optional[Dict] = None,
        extract_relations: bool = True,
        entity_types: Optional[List[str]] = None,
        relation_types: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Process a single document through the entire pipeline.
        
        Args:
            text: Raw document text.
            doc_id: Document identifier.
            metadata: Additional document metadata.
            extract_relations: Whether to extract relations.
            entity_types: Filter for specific entity types.
            relation_types: Filter for specific relation types.
            
        Returns:
            Dictionary with complete extraction results.
        """
        timer = Timer()
        timer.start()
        
        doc_id = doc_id or f"doc_{self.stats['documents_processed'] + 1}"
        logger.info(f"Processing document: {doc_id}")
        
        result = {
            'doc_id': doc_id,
            'metadata': metadata or {},
            'timestamp': datetime.now().isoformat(),
            'status': 'processing'
        }
        
        try:
            # Step 1: Preprocessing
            logger.info(f"[{doc_id}] Step 1/4: Preprocessing...")
            preprocessed = self.preprocessor.process_document(
                text=text,
                doc_id=doc_id,
                metadata=metadata
            )
            result['preprocessing'] = {
                'original_length': preprocessed['original_length'],
                'cleaned_length': preprocessed['cleaned_length'],
                'num_sentences': preprocessed['num_sentences'],
                'num_tokens': preprocessed['num_tokens']
            }
            
            # Step 2: Named Entity Recognition
            logger.info(f"[{doc_id}] Step 2/4: Entity Extraction...")
            ner_results = self.ner_extractor.extract(
                text=preprocessed['cleaned_text'],
                entity_types=entity_types
            )
            result['entities'] = ner_results
            
            # Step 3: Relation Extraction
            if extract_relations and ner_results['total_entities'] >= 2:
                logger.info(f"[{doc_id}] Step 3/4: Relation Extraction...")
                rel_results = self.relation_extractor.extract_relations(
                    text=preprocessed['cleaned_text'],
                    entities=ner_results['entities'],
                    relation_types=relation_types
                )
                result['relations'] = rel_results
            else:
                logger.info(
                    f"[{doc_id}] Step 3/4: Skipping relation extraction "
                    f"({'disabled' if not extract_relations else 'insufficient entities'})."
                )
                result['relations'] = {
                    'relations': [],
                    'total_relations': 0
                }
            
            # Step 4: Post-processing
            logger.info(f"[{doc_id}] Step 4/4: Post-processing...")
            result = self.postprocessor.process(result)
            result['status'] = 'completed'
            
            # Update stats
            self.stats['documents_processed'] += 1
            self.stats['total_entities_extracted'] += (
                result.get('entities', {}).get('total_entities', 0)
            )
            self.stats['total_relations_extracted'] += (
                result.get('relations', {}).get('total_relations', 0)
            )
            
        except Exception as e:
            logger.error(f"[{doc_id}] Pipeline error: {e}", exc_info=True)
            result['status'] = 'error'
            result['error'] = str(e)
            self.stats['errors'].append({
                'doc_id': doc_id,
                'error': str(e)
            })
        
        elapsed = timer.stop()
        result['processing_time_seconds'] = round(elapsed, 3)
        self.stats['total_processing_time'] += elapsed
        
        logger.info(
            f"[{doc_id}] Processing completed in {elapsed:.3f}s. "
            f"Entities: {result.get('entities', {}).get('total_entities', 0)}, "
            f"Relations: {result.get('relations', {}).get('total_relations', 0)}"
        )
        
        return result
    
    def process_batch(
        self,
        documents: List[Dict[str, Any]],
        extract_relations: bool = True,
        entity_types: Optional[List[str]] = None,
        relation_types: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """Process a batch of documents.
        
        Args:
            documents: List of document dicts with 'text', optional
                      'doc_id' and 'metadata'.
            extract_relations: Whether to extract relations.
            entity_types: Filter for specific entity types.
            relation_types: Filter for specific relation types.
            
        Returns:
            List of extraction result dictionaries.
        """
        logger.info(f"Processing batch of {len(documents)} documents.")
        results = []
        
        for i, doc in enumerate(documents):
            logger.info(
                f"Processing document {i + 1}/{len(documents)}"
            )
            result = self.process_document(
                text=doc.get('text', ''),
                doc_id=doc.get('doc_id', f'doc_{i + 1}'),
                metadata=doc.get('metadata'),
                extract_relations=extract_relations,
                entity_types=entity_types,
                relation_types=relation_types
            )
            results.append(result)
        
        logger.info(
            f"Batch processing complete. "
            f"Processed: {len(results)}/{len(documents)}"
        )
        
        return results
    
    def save_results(
        self,
        results: Any,
        output_path: str,
        format: str = 'json'
    ) -> str:
        """Save extraction results to file.
        
        Args:
            results: Extraction results to save.
            output_path: Output file path.
            format: Output format ('json' or 'jsonl').
            
        Returns:
            Path to the saved file.
        """
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        
        if format == 'jsonl':
            with open(output, 'w', encoding='utf-8') as f:
                if isinstance(results, list):
                    for result in results:
                        f.write(json.dumps(result, default=str) + '\n')
                else:
                    f.write(json.dumps(results, default=str) + '\n')
        else:
            with open(output, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, default=str)
        
        logger.info(f"Results saved to: {output}")
        return str(output)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get pipeline processing statistics."""
        stats = dict(self.stats)
        if stats['documents_processed'] > 0:
            stats['avg_processing_time'] = round(
                stats['total_processing_time'] / stats['documents_processed'],
                3
            )
            stats['avg_entities_per_doc'] = round(
                stats['total_entities_extracted'] / stats['documents_processed'],
                1
            )
        return stats
