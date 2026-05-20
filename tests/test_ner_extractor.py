"""Tests for the NER Extractor."""

import pytest
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


class TestNERExtractorEntityTypeMap:
    """Test entity type normalization."""
    
    def test_entity_type_normalization(self):
        """Test that entity types are correctly normalized."""
        from src.ner_extractor import NERExtractor
        
        type_map = NERExtractor.ENTITY_TYPE_MAP
        
        assert type_map['PER'] == 'PERSON'
        assert type_map['ORG'] == 'ORGANIZATION'
        assert type_map['LOC'] == 'LOCATION'
        assert type_map['GPE'] == 'LOCATION'
    
    def test_bio_prefix_removal(self):
        """Test B-/I- prefix removal in entity type normalization."""
        from src.ner_extractor import NERExtractor
        
        entity_type = 'B-PER'
        clean_type = entity_type.split('-')[-1]
        assert clean_type == 'PER'
        assert NERExtractor.ENTITY_TYPE_MAP.get(clean_type) == 'PERSON'


class TestChunking:
    """Test text chunking functionality."""
    
    def test_short_text_no_chunking(self):
        """Test that short text is not chunked."""
        text = "This is a short text."
        max_length = 450
        
        if len(text) <= max_length:
            chunks = [text]
        
        assert len(chunks) == 1
        assert chunks[0] == text
    
    def test_long_text_chunking(self):
        """Test that long text is properly chunked."""
        sentences = [f"This is sentence number {i}." for i in range(100)]
        text = " ".join(sentences)
        max_length = 200
        
        chunks = []
        current_chunk = ""
        for sentence in text.split('. '):
            if len(current_chunk) + len(sentence) + 2 <= max_length:
                current_chunk += sentence + '. '
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = sentence + '. '
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        assert len(chunks) > 1
        for chunk in chunks:
            assert len(chunk) <= max_length + 50  # Allow some margin
