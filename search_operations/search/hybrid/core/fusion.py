"""
Result Fusion Module

This module provides result fusion strategies for combining search results
from multiple retrieval methods (dense vector, sparse vector, keyword search)
into a unified ranked list.
"""

import logging
from typing import List, Dict, Any
from collections import defaultdict

logger = logging.getLogger(__name__)


async def fuse_results_rrf(
    results_list: List[List[Dict[str, Any]]],
    k: int = 60
) -> List[Dict[str, Any]]:
    """
    Fuse results using Reciprocal Rank Fusion (RRF).
    
    RRF is a simple yet effective method for combining rankings from multiple
    sources. It assigns scores based on the reciprocal of the rank position,
    with a constant k added to reduce the impact of high-ranked outliers.
    
    Formula: RRF_score(d) = Σ (1 / (k + rank(d)))
    
    Args:
        results_list: List of result lists from different searches
        k: RRF parameter (typically 60, controls score distribution)
        
    Returns:
        Fused and ranked results with fusion_score added
    """
    if not results_list:
        logger.warning("Empty results list provided to RRF fusion")
        return []
    
    # Dictionary to accumulate RRF scores for each document
    scores: Dict[str, float] = defaultdict(float)
    doc_data: Dict[str, Dict[str, Any]] = {}
    
    # Process each result list
    for results in results_list:
        for rank, doc in enumerate(results, start=1):
            # Use 'id' or 'pk' as document identifier
            doc_id = str(doc.get('id', doc.get('pk', '')))
            
            # Calculate RRF score contribution
            scores[doc_id] += 1.0 / (k + rank)
            
            # Store document data (keep first occurrence)
            if doc_id not in doc_data:
                doc_data[doc_id] = doc
    
    # Sort documents by fused score (descending)
    sorted_docs = sorted(
        scores.items(),
        key=lambda x: x[1],
        reverse=True
    )
    
    # Build final result list with fusion scores
    fused_results = []
    for doc_id, score in sorted_docs:
        doc = doc_data[doc_id].copy()
        doc['fusion_score'] = score
        fused_results.append(doc)
    
    logger.debug(
        f"RRF fusion completed - "
        f"input_lists: {len(results_list)}, "
        f"unique_docs: {len(fused_results)}, "
        f"k: {k}"
    )
    
    return fused_results


async def fuse_results_weighted(
    vector_results: List[Dict[str, Any]],
    sparse_results: List[Dict[str, Any]],
    vector_weight: float,
    sparse_weight: float
) -> List[Dict[str, Any]]:
    """
    Fuse results using weighted scoring.
    
    This method combines results from vector and sparse searches by
    normalizing their scores and applying configurable weights. Documents
    appearing in both result sets get combined scores.
    
    Args:
        vector_results: Results from vector search
        sparse_results: Results from sparse search
        vector_weight: Weight for vector scores (will be normalized)
        sparse_weight: Weight for sparse scores (will be normalized)
        
    Returns:
        Fused and ranked results with fusion_score, vector_score, and sparse_score
    """
    if not vector_results and not sparse_results:
        logger.warning("Empty results provided to weighted fusion")
        return []
    
    # Dictionary to accumulate scores for each document
    scores: Dict[str, Dict[str, float]] = defaultdict(
        lambda: {'vector': 0.0, 'sparse': 0.0}
    )
    doc_data: Dict[str, Dict[str, Any]] = {}
    
    # Normalize weights to sum to 1.0
    total_weight = vector_weight + sparse_weight
    if total_weight <= 0:
        logger.error("Invalid weights: sum must be positive")
        total_weight = 1.0
        vector_weight = 0.5
        sparse_weight = 0.5
    
    norm_vector_weight = vector_weight / total_weight
    norm_sparse_weight = sparse_weight / total_weight
    
    # Process vector results
    for doc in vector_results:
        doc_id = str(doc.get('id', doc.get('pk', '')))
        # Use distance or score field
        scores[doc_id]['vector'] = doc.get('distance', doc.get('score', 0.0))
        doc_data[doc_id] = doc
    
    # Process sparse results
    for doc in sparse_results:
        doc_id = str(doc.get('id', doc.get('pk', '')))
        # Use distance or score field
        scores[doc_id]['sparse'] = doc.get('distance', doc.get('score', 0.0))
        if doc_id not in doc_data:
            doc_data[doc_id] = doc
    
    # Calculate weighted final scores
    final_scores = {}
    for doc_id, score_dict in scores.items():
        final_scores[doc_id] = (
            score_dict['vector'] * norm_vector_weight +
            score_dict['sparse'] * norm_sparse_weight
        )
    
    # Sort by final score (descending)
    sorted_docs = sorted(
        final_scores.items(),
        key=lambda x: x[1],
        reverse=True
    )
    
    # Build final result list with detailed scores
    fused_results = []
    for doc_id, score in sorted_docs:
        doc = doc_data[doc_id].copy()
        doc['fusion_score'] = score
        doc['vector_score'] = scores[doc_id]['vector']
        doc['sparse_score'] = scores[doc_id]['sparse']
        fused_results.append(doc)
    
    logger.debug(
        f"Weighted fusion completed - "
        f"vector_results: {len(vector_results)}, "
        f"sparse_results: {len(sparse_results)}, "
        f"unique_docs: {len(fused_results)}, "
        f"weights: (vector={norm_vector_weight:.2f}, sparse={norm_sparse_weight:.2f})"
    )
    
    return fused_results


def deduplicate_results(
    results: List[Dict[str, Any]],
    id_field: str = 'id'
) -> List[Dict[str, Any]]:
    """
    Remove duplicate documents from results, keeping the first occurrence.
    
    Args:
        results: List of search results
        id_field: Field name to use for identifying duplicates
        
    Returns:
        Deduplicated list of results
    """
    seen_ids = set()
    deduplicated = []
    
    for doc in results:
        doc_id = str(doc.get(id_field, doc.get('pk', '')))
        if doc_id not in seen_ids:
            seen_ids.add(doc_id)
            deduplicated.append(doc)
    
    if len(deduplicated) < len(results):
        logger.debug(
            f"Removed {len(results) - len(deduplicated)} duplicate documents"
        )
    
    return deduplicated


def normalize_scores(
    results: List[Dict[str, Any]],
    score_field: str = 'distance'
) -> List[Dict[str, Any]]:
    """
    Normalize scores in results to [0, 1] range using min-max normalization.
    
    Args:
        results: List of search results
        score_field: Field name containing the score to normalize
        
    Returns:
        Results with normalized scores
    """
    if not results:
        return results
    
    # Extract scores
    scores = [doc.get(score_field, 0.0) for doc in results]
    
    if not scores:
        return results
    
    min_score = min(scores)
    max_score = max(scores)
    
    # Avoid division by zero
    score_range = max_score - min_score
    if score_range == 0:
        # All scores are the same
        for doc in results:
            doc[f'normalized_{score_field}'] = 1.0
        return results
    
    # Normalize each score
    for doc, score in zip(results, scores):
        normalized = (score - min_score) / score_range
        doc[f'normalized_{score_field}'] = normalized
    
    logger.debug(f"Normalized {len(results)} scores to [0, 1] range")
    
    return results

