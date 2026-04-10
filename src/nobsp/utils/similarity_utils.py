#!/usr/bin/env python3
"""
Similarity utilities for NObSP embeddings.

This module provides core functions for similarity search, comparison,
and analysis using NObSP contribution embeddings.
"""

import numpy as np
import torch
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Union
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize


def compute_cosine_similarity(
    query: np.ndarray,
    database: np.ndarray,
    normalize_features: bool = True
) -> np.ndarray:
    """
    Compute cosine similarity between query and database vectors.
    
    Parameters
    ----------
    query : np.ndarray
        Query vector(s) of shape [d] or [n_queries, d]
    database : np.ndarray
        Database vectors of shape [n_samples, d]
    normalize_features : bool
        Whether to L2-normalize features before computing similarity
        
    Returns
    -------
    similarities : np.ndarray
        Similarity scores of shape [n_samples] or [n_queries, n_samples]
    """
    # Ensure 2D arrays
    if query.ndim == 1:
        query = query.reshape(1, -1)
    
    if normalize_features:
        query = normalize(query, norm='l2', axis=1)
        database = normalize(database, norm='l2', axis=1)
    
    # Compute cosine similarity
    similarities = cosine_similarity(query, database)
    
    # Return 1D array if single query
    if similarities.shape[0] == 1:
        return similarities[0]
    return similarities


def find_top_k_similar(
    query_embedding: np.ndarray,
    database_embeddings: np.ndarray,
    k: int = 5,
    exclude_self: bool = True,
    method: str = 'cosine'
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Find top-k most similar samples to query.
    
    Parameters
    ----------
    query_embedding : np.ndarray
        Query embedding vector [d]
    database_embeddings : np.ndarray
        Database of embeddings [n_samples, d]
    k : int
        Number of similar samples to retrieve
    exclude_self : bool
        Whether to exclude exact match (for self-queries)
    method : str
        Similarity metric ('cosine', 'euclidean', 'manhattan')
        
    Returns
    -------
    indices : np.ndarray
        Indices of top-k similar samples [k]
    scores : np.ndarray
        Similarity scores for top-k samples [k]
    """
    if method == 'cosine':
        similarities = compute_cosine_similarity(query_embedding, database_embeddings)
        # Higher is better for cosine
        if exclude_self:
            # Find and exclude exact match
            max_idx = np.argmax(similarities)
            if similarities[max_idx] > 0.9999:  # Near-perfect match
                similarities[max_idx] = -1
        indices = np.argsort(similarities)[::-1][:k]
        scores = similarities[indices]
        
    elif method == 'euclidean':
        # Compute Euclidean distances
        distances = np.linalg.norm(database_embeddings - query_embedding, axis=1)
        # Lower is better for distance
        if exclude_self:
            min_idx = np.argmin(distances)
            if distances[min_idx] < 1e-6:  # Near-zero distance
                distances[min_idx] = np.inf
        indices = np.argsort(distances)[:k]
        scores = -distances[indices]  # Negate for consistency (higher is better)
        
    elif method == 'manhattan':
        # Compute Manhattan distances
        distances = np.sum(np.abs(database_embeddings - query_embedding), axis=1)
        if exclude_self:
            min_idx = np.argmin(distances)
            if distances[min_idx] < 1e-6:
                distances[min_idx] = np.inf
        indices = np.argsort(distances)[:k]
        scores = -distances[indices]
        
    else:
        raise ValueError(f"Unknown similarity method: {method}")
    
    return indices, scores


def compute_contribution_embeddings(
    features: np.ndarray,
    contributions: np.ndarray,
    target_class: Optional[int] = None,
    aggregation: str = 'none'
) -> np.ndarray:
    """
    Compute embeddings from NObSP contributions.
    
    Parameters
    ----------
    features : np.ndarray
        Feature vectors [n_samples, n_features]
    contributions : np.ndarray
        NObSP contributions [n_samples, n_features, n_classes]
    target_class : int, optional
        Specific class to use for embeddings. If None, use predicted class.
    aggregation : str
        How to aggregate multi-class contributions:
        - 'none': Use single class (target or predicted)
        - 'mean': Average across all classes
        - 'max': Max contribution per feature
        - 'weighted': Weight by prediction confidence
        
    Returns
    -------
    embeddings : np.ndarray
        Contribution embeddings [n_samples, n_features]
    """
    n_samples, n_features = features.shape[:2]
    
    if contributions.ndim == 2:
        # Already single-class contributions
        return contributions
    
    if aggregation == 'none':
        # Use specific class
        if target_class is not None:
            embeddings = contributions[:, :, target_class]
        else:
            # Use predicted class (max contribution sum)
            class_scores = np.sum(contributions, axis=1)
            predicted_classes = np.argmax(class_scores, axis=1)
            embeddings = np.array([
                contributions[i, :, predicted_classes[i]] 
                for i in range(n_samples)
            ])
    
    elif aggregation == 'mean':
        embeddings = np.mean(contributions, axis=2)
    
    elif aggregation == 'max':
        embeddings = np.max(contributions, axis=2)
    
    elif aggregation == 'weighted':
        # Weight by softmax of total contributions
        class_scores = np.sum(contributions, axis=1)
        weights = np.exp(class_scores) / np.sum(np.exp(class_scores), axis=1, keepdims=True)
        embeddings = np.sum(contributions * weights[:, np.newaxis, :], axis=2)
    
    else:
        raise ValueError(f"Unknown aggregation method: {aggregation}")
    
    return embeddings


def analyze_channel_importance(
    query_contrib: np.ndarray,
    retrieved_contribs: np.ndarray,
    top_k: int = 10
) -> Dict[str, np.ndarray]:
    """
    Analyze which channels drive similarity between images.
    
    Parameters
    ----------
    query_contrib : np.ndarray
        Query contribution vector [n_features]
    retrieved_contribs : np.ndarray
        Retrieved contribution vectors [n_retrieved, n_features]
    top_k : int
        Number of top channels to identify
        
    Returns
    -------
    analysis : dict
        Dictionary containing:
        - 'shared_positive': Indices of top shared positive channels
        - 'shared_negative': Indices of top shared negative channels
        - 'importance_scores': Channel importance scores
        - 'correlation_per_channel': Correlation between query and retrieved
    """
    # Compute shared positive contributions
    positive_mask = (query_contrib > 0) & (np.mean(retrieved_contribs > 0, axis=0) > 0.5)
    positive_importance = np.abs(query_contrib) * positive_mask
    
    # Compute shared negative contributions
    negative_mask = (query_contrib < 0) & (np.mean(retrieved_contribs < 0, axis=0) > 0.5)
    negative_importance = np.abs(query_contrib) * negative_mask
    
    # Get top channels
    top_positive = np.argsort(positive_importance)[::-1][:top_k]
    top_negative = np.argsort(negative_importance)[::-1][:top_k]
    
    # Compute per-channel consistency (how similar are the contributions across retrieved samples)
    # High consistency means all retrieved samples have similar contribution for that channel
    channel_consistency = []
    for i in range(len(query_contrib)):
        retrieved_vals = retrieved_contribs[:, i]
        # Check if query and retrieved have same sign (positive or negative contribution)
        same_sign = np.sign(query_contrib[i]) == np.sign(np.mean(retrieved_vals))
        # Compute consistency as inverse of std deviation (normalized)
        if np.abs(np.mean(retrieved_vals)) > 0:
            consistency = same_sign * (1 - np.std(retrieved_vals) / (np.abs(np.mean(retrieved_vals)) + 1e-8))
        else:
            consistency = 0.0
        channel_consistency.append(consistency)
    channel_consistency = np.array(channel_consistency)
    
    # Overall importance combining magnitude and consistency
    importance_scores = np.abs(query_contrib) * np.abs(channel_consistency)
    
    return {
        'shared_positive': top_positive,
        'shared_negative': top_negative,
        'importance_scores': importance_scores,
        'correlation_per_channel': channel_consistency,
        'top_important': np.argsort(importance_scores)[::-1][:top_k]
    }


def hybrid_similarity_fusion(
    contrib_similarities: np.ndarray,
    feature_similarities: np.ndarray,
    alpha: float = 0.5,
    normalization: str = 'minmax'
) -> np.ndarray:
    """
    Fuse contribution-based and feature-based similarities.
    
    Parameters
    ----------
    contrib_similarities : np.ndarray
        Similarity scores from contributions [n_samples]
    feature_similarities : np.ndarray
        Similarity scores from raw features [n_samples]
    alpha : float
        Weight for contribution similarity (0 to 1)
        Final = alpha * contrib + (1-alpha) * feature
    normalization : str
        How to normalize scores before fusion:
        - 'minmax': Min-max normalization
        - 'zscore': Z-score normalization
        - 'none': No normalization
        
    Returns
    -------
    fused_similarities : np.ndarray
        Hybrid similarity scores [n_samples]
    """
    if normalization == 'minmax':
        # Min-max normalization to [0, 1]
        def normalize(x):
            if x.max() - x.min() > 1e-8:
                return (x - x.min()) / (x.max() - x.min())
            return x
        contrib_norm = normalize(contrib_similarities)
        feature_norm = normalize(feature_similarities)
        
    elif normalization == 'zscore':
        # Z-score normalization
        def normalize(x):
            if x.std() > 1e-8:
                return (x - x.mean()) / x.std()
            return x
        contrib_norm = normalize(contrib_similarities)
        feature_norm = normalize(feature_similarities)
        
    else:  # 'none'
        contrib_norm = contrib_similarities
        feature_norm = feature_similarities
    
    # Weighted fusion
    fused_similarities = alpha * contrib_norm + (1 - alpha) * feature_norm
    
    return fused_similarities


def compute_similarity_matrix(
    embeddings: np.ndarray,
    method: str = 'cosine',
    symmetric: bool = True
) -> np.ndarray:
    """
    Compute full pairwise similarity matrix.
    
    Parameters
    ----------
    embeddings : np.ndarray
        Embedding vectors [n_samples, n_features]
    method : str
        Similarity metric to use
    symmetric : bool
        Whether to enforce symmetry (for numerical stability)
        
    Returns
    -------
    similarity_matrix : np.ndarray
        Pairwise similarities [n_samples, n_samples]
    """
    n_samples = embeddings.shape[0]
    
    if method == 'cosine':
        # Normalize embeddings
        embeddings_norm = normalize(embeddings, norm='l2', axis=1)
        similarity_matrix = np.dot(embeddings_norm, embeddings_norm.T)
        
    elif method == 'euclidean':
        # Compute pairwise Euclidean distances
        distances = np.zeros((n_samples, n_samples))
        for i in range(n_samples):
            distances[i] = np.linalg.norm(embeddings - embeddings[i], axis=1)
        # Convert to similarity (inverse distance)
        similarity_matrix = 1.0 / (1.0 + distances)
        
    else:
        raise ValueError(f"Unknown method: {method}")
    
    if symmetric:
        # Ensure perfect symmetry
        similarity_matrix = (similarity_matrix + similarity_matrix.T) / 2
        # Set diagonal to 1
        np.fill_diagonal(similarity_matrix, 1.0)
    
    return similarity_matrix


def explain_similarity(
    query_contrib: np.ndarray,
    retrieved_contrib: np.ndarray,
    feature_names: Optional[List[str]] = None,
    top_channels: int = 10
) -> Dict[str, any]:
    """
    Generate explanation for why two samples are similar.
    
    Parameters
    ----------
    query_contrib : np.ndarray
        Query contribution vector [n_features]
    retrieved_contrib : np.ndarray
        Retrieved contribution vector [n_features]
    feature_names : list, optional
        Names for features/channels
    top_channels : int
        Number of top channels to explain
        
    Returns
    -------
    explanation : dict
        Dictionary containing:
        - 'similarity_score': Overall similarity
        - 'top_positive_channels': Most important positive channels
        - 'top_negative_channels': Most important negative channels
        - 'channel_contributions': Individual channel similarity contributions
        - 'explanation_text': Human-readable explanation
    """
    # Compute similarity
    similarity = compute_cosine_similarity(query_contrib, retrieved_contrib.reshape(1, -1))[0]
    
    # Compute per-channel contribution to similarity
    # Cosine similarity = sum(q_i * r_i) / (||q|| * ||r||)
    q_norm = np.linalg.norm(query_contrib)
    r_norm = np.linalg.norm(retrieved_contrib)
    channel_contributions = (query_contrib * retrieved_contrib) / (q_norm * r_norm)
    
    # Find top contributing channels
    top_positive_idx = np.argsort(channel_contributions)[::-1][:top_channels]
    top_negative_idx = np.argsort(channel_contributions)[:top_channels]
    
    # Generate feature names if not provided
    if feature_names is None:
        feature_names = [f"Channel_{i}" for i in range(len(query_contrib))]
    
    # Build explanation
    explanation = {
        'similarity_score': similarity,
        'top_positive_channels': [
            (feature_names[idx], channel_contributions[idx], 
             query_contrib[idx], retrieved_contrib[idx])
            for idx in top_positive_idx if channel_contributions[idx] > 0
        ],
        'top_negative_channels': [
            (feature_names[idx], channel_contributions[idx],
             query_contrib[idx], retrieved_contrib[idx])
            for idx in top_negative_idx if channel_contributions[idx] < 0
        ],
        'channel_contributions': channel_contributions,
    }
    
    # Generate text explanation
    text = f"Similarity Score: {similarity:.3f}\n\n"
    text += "Top Contributing Channels (Positive):\n"
    for name, contrib, q_val, r_val in explanation['top_positive_channels'][:5]:
        text += f"  - {name}: {contrib:.3f} (Q:{q_val:.2f}, R:{r_val:.2f})\n"
    
    if explanation['top_negative_channels']:
        text += "\nTop Contributing Channels (Negative):\n"
        for name, contrib, q_val, r_val in explanation['top_negative_channels'][:3]:
            text += f"  - {name}: {contrib:.3f} (Q:{q_val:.2f}, R:{r_val:.2f})\n"
    
    explanation['explanation_text'] = text
    
    return explanation


def compute_class_statistics(
    embeddings: np.ndarray,
    labels: np.ndarray,
    class_names: Optional[Dict[int, str]] = None
) -> Dict[str, any]:
    """
    Compute per-class statistics for embeddings.
    
    Parameters
    ----------
    embeddings : np.ndarray
        Embedding vectors [n_samples, n_features]
    labels : np.ndarray
        Class labels [n_samples]
    class_names : dict, optional
        Mapping from class indices to names
        
    Returns
    -------
    statistics : dict
        Per-class and overall statistics
    """
    unique_classes = np.unique(labels)
    stats = {}
    
    for class_idx in unique_classes:
        class_mask = labels == class_idx
        class_embeddings = embeddings[class_mask]
        
        # Compute intra-class similarity
        if len(class_embeddings) > 1:
            class_sim_matrix = compute_similarity_matrix(class_embeddings)
            # Exclude diagonal
            n = len(class_embeddings)
            intra_sim = (np.sum(class_sim_matrix) - n) / (n * (n - 1))
        else:
            intra_sim = 1.0
        
        class_name = class_names.get(class_idx, f"Class_{class_idx}") if class_names else f"Class_{class_idx}"
        
        stats[class_name] = {
            'count': len(class_embeddings),
            'mean_embedding': np.mean(class_embeddings, axis=0),
            'std_embedding': np.std(class_embeddings, axis=0),
            'intra_class_similarity': intra_sim
        }
    
    # Compute inter-class similarities
    all_means = np.array([stats[cls]['mean_embedding'] 
                         for cls in stats.keys()])
    inter_class_sim = compute_similarity_matrix(all_means)
    
    # Overall statistics
    overall_stats = {
        'total_samples': len(embeddings),
        'num_classes': len(unique_classes),
        'mean_intra_similarity': np.mean([s['intra_class_similarity'] for s in stats.values()]),
        'inter_class_similarity_matrix': inter_class_sim,
        'mean_inter_similarity': np.mean(inter_class_sim[np.triu_indices_from(inter_class_sim, k=1)])
    }
    
    return {
        'per_class': stats,
        'overall': overall_stats
    }