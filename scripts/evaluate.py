#!/usr/bin/env python3
"""Evaluation Script for the NLP Pipeline.

Computes precision, recall, and F1-score for entity extraction
by comparing pipeline predictions against ground truth annotations.

Usage:
    python scripts/evaluate.py --predictions results/predictions.json --ground-truth data/annotations.json
"""

import sys
import os
import json
import argparse
import logging
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.utils import setup_logging, calculate_metrics

logger = logging.getLogger(__name__)


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate NLP Pipeline extraction results."
    )
    
    parser.add_argument(
        '--predictions', '-p',
        type=str,
        required=True,
        help='Path to prediction results (JSON)'
    )
    parser.add_argument(
        '--ground-truth', '-g',
        type=str,
        required=True,
        help='Path to ground truth annotations (JSON)'
    )
    parser.add_argument(
        '--match-type', '-m',
        type=str,
        choices=['exact', 'partial'],
        default='exact',
        help='Entity matching type (default: exact)'
    )
    parser.add_argument(
        '--output', '-o',
        type=str,
        default=None,
        help='Path to save evaluation report'
    )
    
    return parser.parse_args()


def load_predictions(filepath: str) -> dict:
    """Load prediction results from file."""
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def load_ground_truth(filepath: str) -> dict:
    """Load ground truth annotations from file."""
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def evaluate(
    predictions: list,
    ground_truth: list,
    match_type: str = 'exact'
) -> dict:
    """Evaluate predictions against ground truth.
    
    Args:
        predictions: List of prediction result dicts.
        ground_truth: List of ground truth annotation dicts.
        match_type: Type of entity matching.
        
    Returns:
        Evaluation results dictionary.
    """
    # Build ground truth lookup by doc_id
    gt_lookup = {}
    for gt_doc in ground_truth:
        doc_id = gt_doc.get('doc_id', '')
        gt_lookup[doc_id] = gt_doc
    
    all_true_entities = []
    all_pred_entities = []
    per_type_true = defaultdict(list)
    per_type_pred = defaultdict(list)
    
    for pred_doc in predictions:
        doc_id = pred_doc.get('doc_id', '')
        gt_doc = gt_lookup.get(doc_id, {})
        
        # Extract entity tuples
        pred_entities = [
            (e['text'], e['type'])
            for e in pred_doc.get('entities', {}).get('entities', [])
        ]
        true_entities = [
            (e['text'], e['type'])
            for e in gt_doc.get('entities', [])
        ]
        
        all_true_entities.extend(true_entities)
        all_pred_entities.extend(pred_entities)
        
        # Per-type tracking
        for text, etype in true_entities:
            per_type_true[etype].append((text, etype))
        for text, etype in pred_entities:
            per_type_pred[etype].append((text, etype))
    
    # Overall metrics
    overall = calculate_metrics(
        all_true_entities, all_pred_entities, match_type
    )
    
    # Per-type metrics
    per_type_metrics = {}
    all_types = set(per_type_true.keys()) | set(per_type_pred.keys())
    
    for entity_type in sorted(all_types):
        per_type_metrics[entity_type] = calculate_metrics(
            per_type_true.get(entity_type, []),
            per_type_pred.get(entity_type, []),
            match_type
        )
    
    return {
        'overall': overall,
        'per_type': per_type_metrics,
        'match_type': match_type,
        'num_documents': len(predictions),
        'total_true_entities': len(all_true_entities),
        'total_pred_entities': len(all_pred_entities)
    }


def print_report(eval_results: dict):
    """Print a formatted evaluation report."""
    print("\n" + "=" * 70)
    print("EVALUATION REPORT")
    print("=" * 70)
    print(f"Match Type:     {eval_results['match_type']}")
    print(f"Documents:      {eval_results['num_documents']}")
    print(f"True Entities:  {eval_results['total_true_entities']}")
    print(f"Pred Entities:  {eval_results['total_pred_entities']}")
    print("-" * 70)
    
    overall = eval_results['overall']
    print(f"\nOverall Metrics:")
    print(f"  Precision:  {overall['precision']:.4f}")
    print(f"  Recall:     {overall['recall']:.4f}")
    print(f"  F1-Score:   {overall['f1_score']:.4f}")
    print(f"  TP: {overall['true_positives']}  "
          f"FP: {overall['false_positives']}  "
          f"FN: {overall['false_negatives']}")
    
    print(f"\nPer-Type Metrics:")
    print(f"{'Type':<20} {'Precision':<12} {'Recall':<12} "
          f"{'F1-Score':<12} {'Support':<10}")
    print("-" * 66)
    
    for entity_type, metrics in eval_results['per_type'].items():
        print(
            f"{entity_type:<20} "
            f"{metrics['precision']:<12.4f} "
            f"{metrics['recall']:<12.4f} "
            f"{metrics['f1_score']:<12.4f} "
            f"{metrics['support']:<10}"
        )
    
    print("=" * 70)


def main():
    """Main evaluation entry point."""
    args = parse_args()
    setup_logging(level='INFO')
    
    logger.info("Loading predictions and ground truth...")
    
    predictions = load_predictions(args.predictions)
    ground_truth = load_ground_truth(args.ground_truth)
    
    if isinstance(predictions, dict):
        predictions = [predictions]
    if isinstance(ground_truth, dict):
        ground_truth = [ground_truth]
    
    logger.info("Running evaluation...")
    results = evaluate(predictions, ground_truth, args.match_type)
    
    print_report(results)
    
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2)
        logger.info(f"Report saved to: {args.output}")


if __name__ == '__main__':
    main()
