"""Tests for the Document Preprocessor."""

import pytest
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


class TestDocumentPreprocessor:
    """Test suite for the DocumentPreprocessor class."""
    
    def test_clean_text_removes_null_bytes(self):
        """Test that null bytes are removed from text."""
        from src.preprocessor import DocumentPreprocessor
        preprocessor = DocumentPreprocessor(spacy_model='en_core_web_sm')
        
        text = "Hello\x00World"
        cleaned = preprocessor.clean_text(text)
        assert '\x00' not in cleaned
        assert 'Hello' in cleaned
        assert 'World' in cleaned
    
    def test_clean_text_normalizes_whitespace(self):
        """Test that multiple spaces are normalized."""
        from src.preprocessor import DocumentPreprocessor
        preprocessor = DocumentPreprocessor(spacy_model='en_core_web_sm')
        
        text = "Hello    World"
        cleaned = preprocessor.clean_text(text)
        assert '    ' not in cleaned
    
    def test_clean_text_handles_empty_input(self):
        """Test that empty input returns empty string."""
        from src.preprocessor import DocumentPreprocessor
        preprocessor = DocumentPreprocessor(spacy_model='en_core_web_sm')
        
        assert preprocessor.clean_text("") == ""
        assert preprocessor.clean_text(None) == ""
    
    def test_clean_text_normalizes_line_endings(self):
        """Test that Windows line endings are normalized."""
        from src.preprocessor import DocumentPreprocessor
        preprocessor = DocumentPreprocessor(spacy_model='en_core_web_sm')
        
        text = "Hello\r\nWorld"
        cleaned = preprocessor.clean_text(text)
        assert '\r\n' not in cleaned
        assert '\n' in cleaned


class TestSentenceSegmentation:
    """Test suite for sentence segmentation."""
    
    def test_segment_basic_sentences(self):
        """Test basic sentence segmentation."""
        from src.preprocessor import DocumentPreprocessor
        preprocessor = DocumentPreprocessor(spacy_model='en_core_web_sm')
        
        text = "This is sentence one. This is sentence two. And a third one."
        sentences = preprocessor.segment_sentences(text)
        assert len(sentences) >= 2


class TestTokenization:
    """Test suite for tokenization."""
    
    def test_tokenize_basic(self):
        """Test basic tokenization."""
        from src.preprocessor import DocumentPreprocessor
        preprocessor = DocumentPreprocessor(spacy_model='en_core_web_sm')
        
        text = "Apple Inc. is a technology company."
        tokens = preprocessor.tokenize(text)
        assert len(tokens) > 0
        assert all('text' in t for t in tokens)
        assert all('pos' in t for t in tokens)
