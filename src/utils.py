"""Utility Functions.

Common utility functions for logging setup, configuration loading,
timing, and other shared functionality across the pipeline.
"""

import os
import sys
import json
import time
import logging
from typing import Dict, Any, Optional
from pathlib import Path


def setup_logging(
    level: str = "INFO",
    log_file: Optional[str] = None,
    format_str: Optional[str] = None
) -> None:
    """Configure logging for the pipeline.
    
    Args:
        level: Logging level string.
        log_file: Optional path to log file.
        format_str: Optional custom format string.
    """
    if format_str is None:
        format_str = (
            "%(asctime)s | %(levelname)-8s | "
            "%(name)s:%(funcName)s:%(lineno)d | %(message)s"
        )
    
    handlers = [logging.StreamHandler(sys.stdout)]
    
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file, encoding='utf-8'))
    
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format=format_str,
        handlers=handlers,
        force=True
    )


def load_config(config_path: str) -> Dict[str, Any]:
    """Load pipeline configuration from YAML or JSON file.
    
    Args:
        config_path: Path to configuration file.
        
    Returns:
        Configuration dictionary.
        
    Raises:
        FileNotFoundError: If config file doesn't exist.
        ValueError: If config format is unsupported.
    """
    path = Path(config_path)
    
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    suffix = path.suffix.lower()
    
    if suffix in ('.yaml', '.yml'):
        try:
            import yaml
            with open(path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        except ImportError:
            raise ImportError(
                "PyYAML is required for YAML config files. "
                "Install with: pip install pyyaml"
            )
    elif suffix == '.json':
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    else:
        raise ValueError(f"Unsupported config format: {suffix}")


class Timer:
    """Simple timer utility for performance measurement.
    
    Usage:
        timer = Timer()
        timer.start()
        # ... do work ...
        elapsed = timer.stop()
        print(f"Elapsed: {elapsed:.3f}s")
    """
    
    def __init__(self):
        self._start_time = None
        self._elapsed = 0.0
    
    def start(self) -> 'Timer':
        """Start the timer."""
        self._start_time = time.perf_counter()
        return self
    
    def stop(self) -> float:
        """Stop the timer and return elapsed seconds."""
        if self._start_time is None:
            return 0.0
        self._elapsed = time.perf_counter() - self._start_time
        self._start_time = None
        return self._elapsed
    
    @property
    def elapsed(self) -> float:
        """Get elapsed time without stopping."""
        if self._start_time is not None:
            return time.perf_counter() - self._start_time
        return self._elapsed
    
    def __enter__(self):
        self.start()
        return self
    
    def __exit__(self, *args):
        self.stop()


def calculate_metrics(
    true_entities: list,
    predicted_entities: list,
    match_type: str = 'exact'
) -> Dict[str, float]:
    """Calculate precision, recall, and F1-score for entity extraction.
    
    Args:
        true_entities: List of ground truth entity tuples (text, type).
        predicted_entities: List of predicted entity tuples (text, type).
        match_type: Type of matching - 'exact' or 'partial'.
        
    Returns:
        Dictionary with precision, recall, and F1 scores.
    """
    if match_type == 'exact':
        true_set = set(true_entities)
        pred_set = set(predicted_entities)
        
        tp = len(true_set & pred_set)
        fp = len(pred_set - true_set)
        fn = len(true_set - pred_set)
    else:
        # Partial matching based on text overlap
        from difflib import SequenceMatcher
        tp = 0
        matched_true = set()
        matched_pred = set()
        
        for i, pred in enumerate(predicted_entities):
            for j, true in enumerate(true_entities):
                if j in matched_true:
                    continue
                if pred[1] == true[1]:  # Same type
                    similarity = SequenceMatcher(
                        None, pred[0].lower(), true[0].lower()
                    ).ratio()
                    if similarity >= 0.8:
                        tp += 1
                        matched_true.add(j)
                        matched_pred.add(i)
                        break
        
        fp = len(predicted_entities) - len(matched_pred)
        fn = len(true_entities) - len(matched_true)
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0 else 0.0
    )
    
    return {
        'precision': round(precision, 4),
        'recall': round(recall, 4),
        'f1_score': round(f1, 4),
        'true_positives': tp,
        'false_positives': fp,
        'false_negatives': fn,
        'support': len(true_entities)
    }


def format_entity_table(entities: list) -> str:
    """Format entities as a readable table string.
    
    Args:
        entities: List of entity dictionaries.
        
    Returns:
        Formatted table string.
    """
    if not entities:
        return "No entities found."
    
    header = f"{'Entity':<30} {'Type':<15} {'Confidence':<12} {'Source':<12}"
    separator = '-' * 70
    rows = [header, separator]
    
    for entity in entities:
        rows.append(
            f"{entity.get('text', ''):<30} "
            f"{entity.get('type', ''):<15} "
            f"{entity.get('confidence', 0):<12.4f} "
            f"{entity.get('source', 'N/A'):<12}"
        )
    
    return '\n'.join(rows)
