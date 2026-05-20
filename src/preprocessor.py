"""Document Preprocessing Module.

Handles text cleaning, normalization, sentence segmentation, and tokenization
for incoming documents before they enter the NLP extraction pipeline.
"""

import re
import logging
from typing import Dict, List, Optional, Any

import spacy
from spacy.tokens import Doc

logger = logging.getLogger(__name__)


class DocumentPreprocessor:
    """Preprocesses raw documents for downstream NLP tasks.
    
    Handles text cleaning, normalization, sentence segmentation,
    and tokenization using SpaCy's language models.
    
    Attributes:
        nlp: SpaCy language model instance.
        max_length: Maximum document length for processing.
        custom_patterns: Compiled regex patterns for text cleaning.
    """
    
    # Common noise patterns in enterprise documents
    NOISE_PATTERNS = [
        (r'\x00', ''),                          # Null bytes
        (r'\r\n', '\n'),                        # Windows line endings
        (r'\t+', ' '),                           # Tabs to spaces
        (r' {2,}', ' '),                          # Multiple spaces
        (r'\n{3,}', '\n\n'),                    # Excessive newlines
        (r'[\x80-\x9f]', ''),                   # Control characters
        (r'\u200b|\u200c|\u200d|\ufeff', ''),  # Zero-width chars
    ]
    
    def __init__(
        self,
        spacy_model: str = "en_core_web_trf",
        max_length: int = 1_000_000,
        custom_stop_words: Optional[List[str]] = None,
        disable_components: Optional[List[str]] = None
    ):
        """Initialize the document preprocessor.
        
        Args:
            spacy_model: Name of the SpaCy model to load.
            max_length: Maximum text length for SpaCy processing.
            custom_stop_words: Additional stop words to add.
            disable_components: SpaCy pipeline components to disable.
        """
        self.max_length = max_length
        self.disable_components = disable_components or []
        
        logger.info(f"Loading SpaCy model: {spacy_model}")
        try:
            self.nlp = spacy.load(
                spacy_model,
                disable=self.disable_components
            )
            self.nlp.max_length = max_length
        except OSError:
            logger.warning(
                f"Model '{spacy_model}' not found. "
                f"Falling back to 'en_core_web_sm'."
            )
            self.nlp = spacy.load("en_core_web_sm")
            self.nlp.max_length = max_length
        
        if custom_stop_words:
            for word in custom_stop_words:
                self.nlp.vocab[word].is_stop = True
        
        # Compile noise patterns
        self.compiled_patterns = [
            (re.compile(pattern), replacement)
            for pattern, replacement in self.NOISE_PATTERNS
        ]
        
        logger.info("DocumentPreprocessor initialized successfully.")
    
    def clean_text(self, text: str) -> str:
        """Clean raw text by removing noise and normalizing whitespace.
        
        Args:
            text: Raw input text.
            
        Returns:
            Cleaned text string.
        """
        if not text or not isinstance(text, str):
            return ""
        
        cleaned = text
        for pattern, replacement in self.compiled_patterns:
            cleaned = pattern.sub(replacement, cleaned)
        
        # Normalize unicode characters
        import unicodedata
        cleaned = unicodedata.normalize('NFKC', cleaned)
        
        return cleaned.strip()
    
    def segment_sentences(self, text: str) -> List[str]:
        """Segment text into sentences using SpaCy.
        
        Args:
            text: Input text to segment.
            
        Returns:
            List of sentence strings.
        """
        doc = self.nlp(text)
        sentences = [sent.text.strip() for sent in doc.sents if sent.text.strip()]
        logger.debug(f"Segmented text into {len(sentences)} sentences.")
        return sentences
    
    def tokenize(self, text: str) -> List[Dict[str, Any]]:
        """Tokenize text and extract token-level features.
        
        Args:
            text: Input text to tokenize.
            
        Returns:
            List of token dictionaries with features.
        """
        doc = self.nlp(text)
        tokens = []
        
        for token in doc:
            tokens.append({
                'text': token.text,
                'lemma': token.lemma_,
                'pos': token.pos_,
                'tag': token.tag_,
                'dep': token.dep_,
                'is_stop': token.is_stop,
                'is_punct': token.is_punct,
                'is_alpha': token.is_alpha,
                'shape': token.shape_,
                'idx': token.idx
            })
        
        return tokens
    
    def process_document(
        self,
        text: str,
        doc_id: Optional[str] = None,
        metadata: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """Process a single document through the preprocessing pipeline.
        
        Args:
            text: Raw document text.
            doc_id: Optional document identifier.
            metadata: Optional document metadata.
            
        Returns:
            Dictionary containing preprocessed document data.
        """
        logger.info(f"Processing document: {doc_id or 'unknown'}")
        
        # Clean text
        cleaned_text = self.clean_text(text)
        
        if not cleaned_text:
            logger.warning(f"Document {doc_id} is empty after cleaning.")
            return {
                'doc_id': doc_id,
                'original_length': len(text) if text else 0,
                'cleaned_length': 0,
                'sentences': [],
                'tokens': [],
                'spacy_doc': None,
                'metadata': metadata or {}
            }
        
        # Truncate if necessary
        if len(cleaned_text) > self.max_length:
            logger.warning(
                f"Document {doc_id} exceeds max length "
                f"({len(cleaned_text)} > {self.max_length}). Truncating."
            )
            cleaned_text = cleaned_text[:self.max_length]
        
        # Process with SpaCy
        doc = self.nlp(cleaned_text)
        
        # Extract sentences
        sentences = [sent.text.strip() for sent in doc.sents if sent.text.strip()]
        
        # Extract tokens
        tokens = self.tokenize(cleaned_text)
        
        return {
            'doc_id': doc_id,
            'original_length': len(text),
            'cleaned_length': len(cleaned_text),
            'cleaned_text': cleaned_text,
            'sentences': sentences,
            'num_sentences': len(sentences),
            'tokens': tokens,
            'num_tokens': len(tokens),
            'spacy_doc': doc,
            'metadata': metadata or {}
        }
    
    def process_batch(
        self,
        documents: List[Dict[str, Any]],
        batch_size: int = 100
    ) -> List[Dict[str, Any]]:
        """Process a batch of documents.
        
        Args:
            documents: List of document dictionaries with 'text' and optionally
                      'doc_id' and 'metadata' keys.
            batch_size: Number of documents to process at a time.
            
        Returns:
            List of preprocessed document dictionaries.
        """
        results = []
        total = len(documents)
        
        for i in range(0, total, batch_size):
            batch = documents[i:i + batch_size]
            logger.info(
                f"Processing batch {i // batch_size + 1} "
                f"({len(batch)} documents)"
            )
            
            for doc_dict in batch:
                result = self.process_document(
                    text=doc_dict.get('text', ''),
                    doc_id=doc_dict.get('doc_id'),
                    metadata=doc_dict.get('metadata')
                )
                results.append(result)
        
        logger.info(f"Processed {len(results)}/{total} documents.")
        return results
