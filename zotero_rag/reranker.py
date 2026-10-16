"""Cross-encoder reranking model."""

import logging
import torch
from typing import List, Optional, Tuple
import numpy as np
from sentence_transformers import CrossEncoder

from models import Paragraph, RerankedParagraph

logger = logging.getLogger(__name__)


class Reranker:
    """Handles reranking of retrieved candidates using a cross-encoder model."""
    
    def __init__(self, 
                 model_name: str = 'BAAI/bge-reranker-base',
                 device: str = None,
                 batch_size: int = None):
        """Initialize the reranker.

        Args:
            model_name: Name of the cross-encoder model.
            device: Device to use ('cpu', 'cuda', 'mps'). Auto-detect if None.
            batch_size: Batch size for reranking. If None, auto-detect safe size.
        """
        self.model_name = model_name
        self.device = device or ("mps" if torch.backends.mps.is_available() else "cpu")
        self.batch_size = batch_size  # None means auto-detect
        self.model = CrossEncoder(model_name, device=self.device)
        logger.info(f"Reranker initialized with model {model_name} on {self.device}")

    def adaptive_rerank_threshold(self,
                                  rerank_scores: np.ndarray,
                                  base_threshold: Optional[float] = None,
                                  mode: str = 'medium') -> float:
        """Adjust rerank threshold based on score distribution.
        
        Args:
            rerank_scores: Array of rerank probability scores (0-1).
            base_threshold: Required threshold for custom mode; unused otherwise.
            mode: Adaptive mode ('low', 'medium', 'high') or 'custom'.
            
        Returns:
            Adjusted threshold value.
        """
        if mode == 'custom':
            if base_threshold is None:
                raise ValueError("A rerank threshold is required in custom mode")
            return base_threshold

        std_multipliers = {'low': -2, 'medium': -1, 'high': 1}
        if mode not in std_multipliers:
            raise ValueError(f"Unsupported rerank threshold mode: {mode}")

        # Calculate statistics
        mean_score = np.mean(rerank_scores)
        std_score = np.std(rerank_scores)
        max_score = np.max(rerank_scores)

        adjusted_threshold = mean_score + std_multipliers[mode] * std_score
        logger.debug(
            f"Adaptive threshold ({mode}): {adjusted_threshold:.3f} "
            f"(mean={mean_score:.3f}, std={std_score:.3f}, max={max_score:.3f})"
        )
        return float(adjusted_threshold)
    
    def _find_safe_batch_size(self, 
                            pairs: List[List[str]], 
                            start_size: int = 2, 
                            max_size: int = 128,
                            target_memory_fraction: float = 0.75) -> int:
        """Find safe batch size targeting specific memory usage.
        
        Args:
            pairs: Sample of text pairs to test with.
            start_size: Initial batch size to try.
            max_size: Maximum batch size to test.
            target_memory_fraction: Target fraction of memory to use (0.0-1.0).
            
        Returns:
            Safe batch size.
        """
        import torch
        test_sample = pairs[:min(100, len(pairs))]
        
        current_size = start_size
        last_safe_size = start_size
        
        while current_size <= max_size:
            try:
                _ = self.model.predict(test_sample[:current_size], show_progress_bar=False)
                last_safe_size = current_size
                # Scale up more aggressively to find limit
                current_size = int(current_size * 1.5)
            except (RuntimeError, torch.cuda.OutOfMemoryError) as e:
                error_str = str(e).lower()
                if any(phrase in error_str for phrase in 
                      ["out of memory", "buffer size", "mps", "cuda", "memory"]):
                    # Hit OOM, scale back to target fraction
                    return max(start_size, int(last_safe_size * target_memory_fraction))
                else:
                    return last_safe_size
            except Exception:
                return last_safe_size
        
        # Hit max size without OOM, use target fraction of max
        return max(start_size, int(last_safe_size * target_memory_fraction))
    
    def rerank(self,
            query: str,
            candidates: List[Tuple[Paragraph, float]],
            threshold: Optional[float] = None,
            threshold_mode: str = 'medium',
            progress_callback=None,
            query_variations: List[str] = None) -> List[RerankedParagraph]:
        """Rerank candidates using cross-encoder scores.
        
        Args:
            query: The query string.
            candidates: List of (Paragraph, retrieval_score) tuples.
            threshold: Exact custom score threshold; None for adaptive modes.
            threshold_mode: Adaptive threshold mode or 'custom' for the exact threshold.
            progress_callback: Function(current, total, message) for progress updates.
            query_variations: List of query paraphrases.
            
        Returns:
            List of RerankedParagraph items sorted by rerank_score descending.
        """
        if not candidates:
            return []
        
        # Use query variations if provided, otherwise just use the original query
        queries_to_use = query_variations if query_variations else [query]
        
        # Prepare pairs for the reranker
        pairs = [[query, p[0].text] for p in candidates]
        
        # Determine batch size
        if self.batch_size is None:
            # Auto-detect safe batch size
            if progress_callback:
                progress_callback(0, len(candidates), "Auto-detecting safe batch size for merged candidates...")
            effective_batch_size = self._find_safe_batch_size(pairs, start_size=2, max_size=128)
            logger.info(f"Auto-detected reranker batch size: {effective_batch_size}")
        else:
            effective_batch_size = self.batch_size
        
        # Score candidates with each query variation and average
        all_probs_per_variation = []
        
        total_operations = len(queries_to_use) * len(candidates)
        completed_operations = 0
        
        for var_idx, q_var in enumerate(queries_to_use):
            # Prepare pairs for this variation
            var_pairs = [[q_var, p[0].text] for p in candidates]
            
            # Predict scores in batches with progress tracking
            all_scores = []
            num_batches = (len(var_pairs) + effective_batch_size - 1) // effective_batch_size
            
            for batch_idx, i in enumerate(range(0, len(var_pairs), effective_batch_size)):
                batch_pairs = var_pairs[i:i + effective_batch_size]
                
                batch_scores = self.model.predict(batch_pairs, activation_fn=torch.nn.Sigmoid(), show_progress_bar=False)
                all_scores.extend(batch_scores)
                
                # Update progress with current variation info
                processed_in_batch = min(len(batch_pairs), len(var_pairs) - i)
                completed_operations += processed_in_batch
                
                if progress_callback:
                    variation_info = f"Paraphrase {var_idx + 1}/{len(queries_to_use)}" if len(queries_to_use) > 1 else ""
                    batch_info = f"Batch {batch_idx + 1}/{num_batches}"
                    message = f"{variation_info} - {batch_info}" if variation_info else batch_info
                    progress_callback(completed_operations, total_operations, message)
            
            var_probs = np.asarray(all_scores)
            all_probs_per_variation.append(var_probs)
        
        # Calculate max probabilities across all variations
        probs = np.max(all_probs_per_variation, axis=0) if len(all_probs_per_variation) > 1 else all_probs_per_variation[0]
        
        # Adapt threshold based on score distribution
        adjusted_threshold = self.adaptive_rerank_threshold(probs, threshold, threshold_mode)
        
        # Combine probabilities with candidate data
        # Each item: (paragraph, retrieval_score, rerank_score)
        scored_candidates = [
            RerankedParagraph(
                paragraph=p[0],
                retrieval_score=p[1],
                rerank_score=float(prob),
            )
            for p, prob in zip(candidates, probs)
        ]
        
        # Filter by adjusted threshold
        filtered_candidates = [
            item for item in scored_candidates
            if item.rerank_score >= adjusted_threshold
        ]
        
        # Sort by rerank score descending
        filtered_candidates.sort(key=lambda x: x.rerank_score, reverse=True)
        
        if progress_callback:
            progress_callback(len(candidates), len(candidates),
                            f"Reranking complete: {len(filtered_candidates)} paragraphs passed threshold.")
        
        # Log threshold statistics
        if threshold_mode == 'custom':
            logger.info(f"Reranking: {len(candidates)} -> {len(filtered_candidates)} paragraphs "
                       f"(custom threshold: {adjusted_threshold:.3f})")
        else:
            logger.info(f"Reranking: {len(candidates)} -> {len(filtered_candidates)} paragraphs "
                       f"(adaptive mode: {threshold_mode}, threshold: {adjusted_threshold:.3f})")
        
        return filtered_candidates
