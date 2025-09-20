"""
Evaluation metrics for the Movie Recommendation System.
"""

import numpy as np
from sklearn.metrics import mean_squared_error, precision_score, recall_score
from typing import List, Tuple

class Evaluator:
    @staticmethod
    def calculate_rmse(true_ratings: np.ndarray, predicted_ratings: np.ndarray) -> float:
        """
        Calculate Root Mean Square Error.
        
        Args:
            true_ratings (np.ndarray): Array of actual ratings
            predicted_ratings (np.ndarray): Array of predicted ratings
            
        Returns:
            float: RMSE value
        """
        return np.sqrt(mean_squared_error(true_ratings, predicted_ratings))

    @staticmethod
    def calculate_precision_at_k(actual_items: List, recommended_items: List, k: int) -> float:
        """
        Calculate Precision@K.
        
        Args:
            actual_items (List): List of items actually relevant to the user
            recommended_items (List): List of recommended items
            k (int): Number of recommendations to consider
            
        Returns:
            float: Precision@K value
        """
        recommended_k = set(recommended_items[:k])
        relevant = set(actual_items)
        
        if len(recommended_k) == 0:
            return 0.0
            
        return len(recommended_k.intersection(relevant)) / len(recommended_k)

    @staticmethod
    def calculate_recall_at_k(actual_items: List, recommended_items: List, k: int) -> float:
        """
        Calculate Recall@K.
        
        Args:
            actual_items (List): List of items actually relevant to the user
            recommended_items (List): List of recommended items
            k (int): Number of recommendations to consider
            
        Returns:
            float: Recall@K value
        """
        recommended_k = set(recommended_items[:k])
        relevant = set(actual_items)
        
        if len(relevant) == 0:
            return 0.0
            
        return len(recommended_k.intersection(relevant)) / len(relevant)

    @staticmethod
    def calculate_ndcg(actual_ratings: List[Tuple[str, float]], 
                      predicted_ratings: List[Tuple[str, float]], 
                      k: int) -> float:
        """
        Calculate Normalized Discounted Cumulative Gain.
        
        Args:
            actual_ratings (List[Tuple[str, float]]): List of (item, rating) pairs for actual ratings
            predicted_ratings (List[Tuple[str, float]]): List of (item, rating) pairs for predicted ratings
            k (int): Number of recommendations to consider
            
        Returns:
            float: NDCG value
        """
        def dcg(ratings: List[Tuple[str, float]], k: int) -> float:
            """Calculate Discounted Cumulative Gain."""
            if not ratings:
                return 0.0
                
            ratings = ratings[:k]
            return ratings[0][1] + sum(r[1] / np.log2(i + 2) for i, r in enumerate(ratings[1:], 1))

        predicted_dcg = dcg(predicted_ratings, k)
        ideal_dcg = dcg(sorted(actual_ratings, key=lambda x: x[1], reverse=True), k)
        
        if ideal_dcg == 0:
            return 0.0
            
        return predicted_dcg / ideal_dcg