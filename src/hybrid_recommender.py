"""
Enhanced recommendation model implementing hybrid filtering
"""

import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MultiLabelBinarizer
from typing import List, Tuple, Dict, Optional

class HybridRecommender:
    def __init__(self, collaborative_weight: float = 0.7):
        """
        Initialize the hybrid recommender.
        
        Args:
            collaborative_weight: Weight for collaborative filtering (1 - this = content weight)
        """
        self.collaborative_weight = collaborative_weight
        self.user_movie_matrix = None
        self.movie_genre_matrix = None
        self.user_similarity_matrix = None
        self.movie_similarity_matrix = None
        self.user_biases = None
        self.movie_biases = None
        self.global_mean = None

    def _prepare_genre_features(self, movies_df: pd.DataFrame) -> np.ndarray:
        """
        Convert genre strings to binary feature matrix.
        
        Args:
            movies_df: DataFrame with movieId and genres columns
            
        Returns:
            Binary genre feature matrix
        """
        # Split genre strings and create binary features
        mlb = MultiLabelBinarizer()
        genres_split = movies_df['genres'].str.split('|')
        genre_features = mlb.fit_transform(genres_split)
        
        # Store movie titles for reference
        self.movie_ids = movies_df.index
        self.movie_id_to_title = movies_df['title'].to_dict()
        self.title_to_movie_id = {v: k for k, v in self.movie_id_to_title.items()}
        
        return genre_features

    def _calculate_biases(self):
        """Calculate global mean, user biases, and movie biases."""
        # Global mean rating
        self.global_mean = np.nanmean(self.user_movie_matrix.values)
        
        # User biases
        user_means = self.user_movie_matrix.mean(axis=1)
        self.user_biases = user_means - self.global_mean
        
        # Movie biases
        movie_means = self.user_movie_matrix.mean(axis=0)
        self.movie_biases = movie_means - self.global_mean

    def fit(self, user_movie_matrix: pd.DataFrame, movies_df: pd.DataFrame):
        """
        Fit the recommendation model.
        
        Args:
            user_movie_matrix: User-movie rating matrix
            movies_df: DataFrame with movie information including genres
        """
        self.user_movie_matrix = user_movie_matrix
        
        # Calculate rating biases
        self._calculate_biases()
        
        # Calculate user similarity matrix
        normalized_ratings = self.user_movie_matrix.fillna(0).values
        self.user_similarity_matrix = pd.DataFrame(
            cosine_similarity(normalized_ratings),
            index=user_movie_matrix.index,
            columns=user_movie_matrix.index
        )
        
        # Calculate movie similarity matrix based on genres
        self.movie_genre_matrix = self._prepare_genre_features(movies_df)
        movie_genre_sim = cosine_similarity(self.movie_genre_matrix)
        self.movie_similarity_matrix = pd.DataFrame(
            movie_genre_sim,
            index=movies_df.index,
            columns=movies_df.index
        )

    def predict_rating(self, user_id: int, movie_title: str) -> float:
        """
        Predict rating for a user-movie pair using hybrid approach.
        
        Args:
            user_id: User ID
            movie_title: Movie title
            
        Returns:
            Predicted rating
        """
        if self.user_movie_matrix is None:
            raise ValueError("Model must be fitted first")
            
        # Get predictions from both models
        cf_pred = self._collaborative_predict(user_id, movie_title)
        cb_pred = self._content_predict(user_id, movie_title)
        
        # Combine predictions
        if cf_pred is None and cb_pred is None:
            return self.global_mean + self.user_biases[user_id] + self.movie_biases[movie_title]
        elif cf_pred is None:
            return cb_pred
        elif cb_pred is None:
            return cf_pred
        else:
            return (self.collaborative_weight * cf_pred + 
                   (1 - self.collaborative_weight) * cb_pred)

    def _collaborative_predict(self, user_id: int, movie_title: str) -> Optional[float]:
        """Predict rating using collaborative filtering."""
        similar_users = self.user_similarity_matrix[user_id].sort_values(ascending=False)[1:6]
        
        if len(similar_users) == 0:
            return None
            
        numerator = 0
        denominator = 0
        
        for other_user, similarity in similar_users.items():
            rating = self.user_movie_matrix.loc[other_user, movie_title]
            if not pd.isna(rating):
                # Adjust rating by user and movie biases
                adjusted_rating = (rating - self.global_mean - 
                                self.user_biases[other_user] - 
                                self.movie_biases[movie_title])
                numerator += similarity * adjusted_rating
                denominator += abs(similarity)
        
        if denominator == 0:
            return None
            
        # Add biases back
        return (self.global_mean + 
                self.user_biases[user_id] + 
                self.movie_biases[movie_title] + 
                numerator / denominator)

    def _content_predict(self, user_id: int, movie_title: str) -> Optional[float]:
        """Predict rating using content-based filtering."""
        # Get user's rated movies
        user_ratings = self.user_movie_matrix.loc[user_id].dropna()
        
        if len(user_ratings) == 0:
            return None
            
        numerator = 0
        denominator = 0
        
        movie_id = self.title_to_movie_id[movie_title]
        for rated_title, rating in user_ratings.items():
            rated_id = self.title_to_movie_id[rated_title]
            similarity = self.movie_similarity_matrix.loc[movie_id, rated_id]
            adjusted_rating = (rating - self.global_mean - 
                            self.user_biases[user_id] - 
                            self.movie_biases[rated_title])
            numerator += similarity * adjusted_rating
            denominator += abs(similarity)
        
        if denominator == 0:
            return None
            
        return (self.global_mean + 
                self.user_biases[user_id] + 
                self.movie_biases[movie_title] + 
                numerator / denominator)

    def get_similar_users(self, user_id: int, n: int = 5) -> pd.Series:
        """
        Find users similar to the given user.
        
        Args:
            user_id: Target user ID
            n: Number of similar users to return
            
        Returns:
            Series of similar users with their similarity scores
        """
        if self.user_similarity_matrix is None:
            raise ValueError("Model must be fitted first")
        
        return self.user_similarity_matrix[user_id].sort_values(ascending=False)[1:n+1]

    def recommend_movies(self, user_id: int, top_n: int = 5) -> List[Tuple[str, float]]:
        """
        Generate movie recommendations for a user.
        
        Args:
            user_id: User ID to generate recommendations for
            top_n: Number of recommendations to generate
            
        Returns:
            List of (movie_title, predicted_rating) tuples
        """
        if self.user_movie_matrix is None:
            raise ValueError("Model must be fitted first")
            
        # Get unwatched movies
        user_ratings = self.user_movie_matrix.loc[user_id]
        unwatched = user_ratings[user_ratings.isna()].index
        
        # Generate predictions
        predictions = []
        for movie in unwatched:
            pred_rating = self.predict_rating(user_id, movie)
            predictions.append((movie, pred_rating))
        
        # Sort and return top N
        return sorted(predictions, key=lambda x: x[1], reverse=True)[:top_n]