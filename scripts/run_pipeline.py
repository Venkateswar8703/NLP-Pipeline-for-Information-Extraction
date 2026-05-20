#!/usr/bin/env python3
"""Run the NLP Pipeline on documents.

Command-line interface for running the structured information
extraction pipeline on documents or document collections.

Usage:
    python scripts/run_pipeline.py --input data/sample/sample_documents.json
    python scripts/run_pipeline.py --input data/sample/sample_documents.json --output results/
    python scripts/run_pipeline.py --input data/sample/sample_documents.json --config config/config.yaml
"""

import sys
import os
import json
import argparse
import logging
from pathlib import Path
from datetime import datetime

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.pipeline import NLPPipeline
from src.utils import setup_logging, Timer

logger = logging.getLogger(__name__)


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "NLP Pipeline for Structured Information Extraction. "
            "Processes documents using transformer-based models for "
            "entity and relation extraction."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        '--input', '-i',
        type=str,
        required=True,
        help='Path to input document(s) (JSON or JSONL format)'
    )
    parser.add_argument(
        '--output', '-o',
        type=str,
        default='./output',
        help='Output directory for results (default: ./output)'
    )
    parser.add_argument(
        '--config', '-c',
        type=str,
        default=None,
        help='Path to pipeline configuration file (YAML/JSON)'
    )
    parser.add_argument(
        '--format', '-f',
        type=str,
        choices=['json', 'jsonl'],
        default='json',
        help='Output format (default: json)'
    )
    parser.add_argument(
        '--no-relations',
        action='store_true',
        help='Skip relation extraction'
    )
    parser.add_argument(
        '--entity-types',
        type=str,
        nargs='+',
        default=None,
        help='Filter for specific entity types (e.g., PERSON ORGANIZATION)'
    )
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging'
    )
    
    return parser.parse_args()


def load_documents(input_path: str) -> list:
    """Load documents from file.
    
    Args:
        input_path: Path to input file.
        
    Returns:
        List of document dictionaries.
    """
    path = Path(input_path)
    
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")
    
    if path.suffix == '.jsonl':
        documents = []
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    documents.append(json.loads(line))
        return documents
    elif path.suffix == '.json':
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            if isinstance(data, list):
                return data
            else:
                return [data]
    else:
        # Try to read as plain text
        with open(path, 'r', encoding='utf-8') as f:
            text = f.read()
        return [{'doc_id': path.stem, 'text': text}]


def main():
    """Main entry point for the pipeline CLI."""
    args = parse_args()
    
    # Setup logging
    log_level = 'DEBUG' if args.verbose else 'INFO'
    setup_logging(level=log_level)
    
    logger.info("=" * 60)
    logger.info("NLP Pipeline for Structured Information Extraction")
    logger.info("=" * 60)
    
    timer = Timer()
    timer.start()
    
    try:
        # Load documents
        logger.info(f"Loading documents from: {args.input}")
        documents = load_documents(args.input)
        logger.info(f"Loaded {len(documents)} document(s).")
        
        # Initialize pipeline
        logger.info("Initializing NLP Pipeline...")
        pipeline = NLPPipeline(config_path=args.config)
        
        # Process documents
        results = pipeline.process_batch(
            documents=documents,
            extract_relations=not args.no_relations,
            entity_types=args.entity_types
        )
        
        # Save results
        output_dir = Path(args.output)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_file = output_dir / f"extraction_results_{timestamp}.{args.format}"
        
        pipeline.save_results(
            results=results,
            output_path=str(output_file),
            format=args.format
        )
        
        # Print summary
        stats = pipeline.get_stats()
        elapsed = timer.stop()
        
        logger.info("\n" + "=" * 60)
        logger.info("PIPELINE EXECUTION SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Documents Processed:    {stats['documents_processed']}")
        logger.info(f"Total Entities:         {stats['total_entities_extracted']}")
        logger.info(f"Total Relations:        {stats['total_relations_extracted']}")
        logger.info(f"Total Time:             {elapsed:.3f}s")
        if stats['documents_processed'] > 0:
            logger.info(
                f"Avg Time/Document:      "
                f"{stats.get('avg_processing_time', 0):.3f}s"
            )
            logger.info(
                f"Avg Entities/Document:  "
                f"{stats.get('avg_entities_per_doc', 0):.1f}"
            )
        logger.info(f"Errors:                 {len(stats['errors'])}")
        logger.info(f"Results saved to:       {output_file}")
        logger.info("=" * 60)
        
    except Exception as e:
        logger.error(f"Pipeline execution failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == '__main__':
    main()
