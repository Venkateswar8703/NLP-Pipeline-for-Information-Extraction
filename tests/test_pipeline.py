"""Tests for the main NLP Pipeline."""

import pytest
from unittest.mock import MagicMock, patch

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


class TestNLPPipeline:
    """Test suite for the NLPPipeline class."""
    
    def test_default_config(self):
        """Test that default configuration is generated correctly."""
        from src.pipeline import NLPPipeline
        config = NLPPipeline._default_config()
        
        assert 'preprocessor' in config
        assert 'ner' in config
        assert 'relation_extraction' in config
        assert 'postprocessor' in config
        assert 'logging' in config
    
    def test_config_keys(self):
        """Test that all required config keys are present."""
        from src.pipeline import NLPPipeline
        config = NLPPipeline._default_config()
        
        assert config['ner']['model_name'] == 'dslim/bert-base-NER'
        assert config['ner']['confidence_threshold'] == 0.85
        assert config['preprocessor']['max_length'] == 1_000_000


class TestPipelineStats:
    """Test suite for pipeline statistics tracking."""
    
    def test_initial_stats(self):
        """Test initial statistics values."""
        stats = {
            'documents_processed': 0,
            'total_entities_extracted': 0,
            'total_relations_extracted': 0,
            'total_processing_time': 0.0,
            'errors': []
        }
        
        assert stats['documents_processed'] == 0
        assert stats['total_entities_extracted'] == 0
        assert len(stats['errors']) == 0
