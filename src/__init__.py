"""NLP Pipeline for Structured Information Extraction.

A production-grade transformer-based NLP pipeline for extracting structured
information from unstructured documents using Transformers and SpaCy.
"""

__version__ = "1.0.0"
__author__ = "Venkateswar"

from src.pipeline import NLPPipeline
from src.preprocessor import DocumentPreprocessor
from src.ner_extractor import NERExtractor
from src.relation_extractor import RelationExtractor
from src.postprocessor import PostProcessor
