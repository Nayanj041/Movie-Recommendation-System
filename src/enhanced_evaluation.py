"""
Enhanced evaluation metrics for recommendation system
"""

import numpy as np
from typing import List, Tuple, Dict
from sklearn.metrics import mean_squared_error, precision_score, recall_score, ndcg_score

class EnhancedEvaluator:
    @staticmethod
    def calculate_rmse(true_ratings: np.ndarray, predicted_ratings: np.ndarray) -> float:
        """
        Calculate Root Mean Square Error.
        
        Args:
            true_ratings: Array of actual ratings
            predicted_ratings: Array of predicted ratings
            
        Returns:
            RMSE value
        """
        return np.sqrt(mean_squared_error(true_ratings, predicted_ratings))

    @staticmethod
    def calculate_mae(true_ratings: np.ndarray, predicted_ratings: np.ndarray) -> float:
        """
        Calculate Mean Absolute Error.
        
        Args:
            true_ratings: Array of actual ratings
            predicted_ratings: Array of predicted ratings
            
        Returns:
            MAE value
        """
        return np.mean(np.abs(true_ratings - predicted_ratings))

    @staticmethod
    def precision_at_k(actual_items: List, recommended_items: List, k: int) -> float:
        """
        Calculate Precision@K.
        
        Args:
            actual_items: List of items actually relevant to the user
            recommended_items: List of recommended items
            k: Number of recommendations to consider
            
        Returns:
            Precision@K value
        """
        if len(recommended_items) == 0:
            return 0.0
        
        recommended_k = set(recommended_items[:k])
        relevant = set(actual_items)
        
        return len(recommended_k.intersection(relevant)) / k

    @staticmethod
    def recall_at_k(actual_items: List, recommended_items: List, k: int) -> float:
        """
        Calculate Recall@K.
        
        Args:
            actual_items: List of items actually relevant to the user
            recommended_items: List of recommended items
            k: Number of recommendations to consider
            
        Returns:
            Recall@K value
        """
        if len(actual_items) == 0:
            return 0.0
        
        recommended_k = set(recommended_items[:k])
        relevant = set(actual_items)
        
        return len(recommended_k.intersection(relevant)) / len(relevant)

    @staticmethod
    def ndcg_at_k(true_ratings: Dict[str, float], 
                  predicted_ratings: Dict[str, float], 
                  k: int) -> float:
        """
        Calculate Normalized Discounted Cumulative Gain@K.
        
        Args:
            true_ratings: Dictionary of true item ratings
            predicted_ratings: Dictionary of predicted item ratings
            k: Number of recommendations to consider
            
        Returns:
            NDCG@K value
        """
        # Get all items
        all_items = list(set(true_ratings.keys()) | set(predicted_ratings.keys()))
        
        # Create arrays for true and predicted ratings
        y_true = np.array([true_ratings.get(item, 0) for item in all_items])
        y_pred = np.array([predicted_ratings.get(item, 0) for item in all_items])
        
        return ndcg_score([y_true], [y_pred], k=k)

    @staticmethod
    def coverage(recommended_items: List[str], all_items: List[str]) -> float:
        """
        Calculate catalog coverage of recommendations.
        
        Args:
            recommended_items: List of all items that were recommended
            all_items: List of all available items
            
        Returns:
            Coverage ratio
        """
        return len(set(recommended_items)) / len(all_items)

    @staticmethod
    def diversity(recommended_items: List[Tuple[str, List[str]]], k: int) -> float:
        """
        Calculate diversity of recommendations based on genres.
        
        Args:
            recommended_items: List of (item_id, genres) tuples
            k: Number of recommendations to consider
            
        Returns:
            Diversity score
        """
        if len(recommended_items) < k:
            return 0.0
            
        # Consider only top-k items
        top_k = recommended_items[:k]
        
        # Calculate pairwise genre differences
        total_diff = 0
        comparisons = 0
        
        for i in range(k):
            for j in range(i + 1, k):
                item1_genres = set(top_k[i][1])
                item2_genres = set(top_k[j][1])
                
                # Jaccard distance as diversity measure
                intersection = len(item1_genres & item2_genres)
                union = len(item1_genres | item2_genres)
                
                if union > 0:
                    similarity = intersection / union
                    total_diff += (1 - similarity)  # Convert similarity to distance
                    comparisons += 1
        
        if comparisons == 0:
            return 0.0
            
        return total_diff / comparisons