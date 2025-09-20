"""
Recommendation model module implementing collaborative filtering.
"""

import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Tuple

class RecommendationModel:
    def __init__(self):
        """Initialize the RecommendationModel."""
        self.similarity_matrix = None
        self.user_movie_matrix = None

    def fit(self, user_movie_matrix: pd.DataFrame) -> None:
        """
        Fit the recommendation model using user-movie matrix.
        
        Args:
            user_movie_matrix (pd.DataFrame): Preprocessed user-movie matrix
        """
        self.user_movie_matrix = user_movie_matrix
        self.similarity_matrix = pd.DataFrame(
            cosine_similarity(user_movie_matrix),
            index=user_movie_matrix.index,
            columns=user_movie_matrix.index
        )

    def get_similar_users(self, user_id: int, n: int = 5) -> pd.Series:
        """
        Find users similar to the given user.
        
        Args:
            user_id (int): Target user ID
            n (int): Number of similar users to return
            
        Returns:
            pd.Series: Similar users with their similarity scores
        """
        if self.similarity_matrix is None:
            raise ValueError("Model must be fitted first")
            
        return self.similarity_matrix[user_id].sort_values(ascending=False)[1:n+1]

    def recommend_movies(self, user_id: int, top_n: int = 5) -> List[Tuple[str, float]]:
        """
        Generate movie recommendations for a user.
        
        Args:
            user_id (int): User ID to generate recommendations for
            top_n (int): Number of recommendations to generate
            
        Returns:
            List[Tuple[str, float]]: List of (movie_title, predicted_rating) tuples
        """
        if self.similarity_matrix is None or self.user_movie_matrix is None:
            raise ValueError("Model must be fitted first")

        # Get similar users
        similar_users = self.get_similar_users(user_id)
        
        # Initialize recommendations
        recommendations = pd.Series(dtype=float)
        
        # Get recommendations from similar users
        for other_user, similarity in similar_users.items():
            other_ratings = self.user_movie_matrix.loc[other_user]
            recommendations = recommendations.add(
                other_ratings * similarity,
                fill_value=0
            )
        
        # Remove already rated movies
        user_ratings = self.user_movie_matrix.loc[user_id]
        recommendations = recommendations[user_ratings.isna()]
        
        # Get top N recommendations
        top_recommendations = recommendations.sort_values(ascending=False).head(top_n)
        
        return list(zip(top_recommendations.index, top_recommendations.values))

    def get_user_preferences(self, user_id: int) -> pd.Series:
        """
        Get a user's movie preferences (ratings).
        
        Args:
            user_id (int): User ID
            
        Returns:
            pd.Series: User's movie ratings
        """
        if self.user_movie_matrix is None:
            raise ValueError("Model must be fitted first")
            
        return self.user_movie_matrix.loc[user_id].dropna().sort_values(ascending=False)

    def get_movie_ratings(self, movie_title: str) -> pd.Series:
        """
        Get all ratings for a specific movie.
        
        Args:
            movie_title (str): Title of the movie
            
        Returns:
            pd.Series: All user ratings for the movie
        """
        if self.user_movie_matrix is None:
            raise ValueError("Model must be fitted first")
            
        return self.user_movie_matrix[movie_title].dropna().sort_values(ascending=False)