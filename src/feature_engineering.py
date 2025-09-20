"""
Feature engineering module for Movie Recommendation System.
Handles advanced feature creation and transformations.
"""

import pandas as pd
import numpy as np
from sklearn.decomposition import TruncatedSVD
from typing import Tuple

class FeatureEngineering:
    def __init__(self):
        """Initialize the FeatureEngineering class."""
        self.svd = None

    def normalize_per_user(self, matrix: pd.DataFrame) -> pd.DataFrame:
        """
        Normalize ratings on a per-user basis.
        
        Args:
            matrix (pd.DataFrame): User-movie matrix
            
        Returns:
            pd.DataFrame: Matrix with normalized ratings per user
        """
        user_mean = matrix.mean(axis=1)
        return matrix.sub(user_mean, axis=0)

    def encode_genres(self, movies_df: pd.DataFrame) -> pd.DataFrame:
        """
        Create one-hot encoded features for movie genres.
        
        Args:
            movies_df (pd.DataFrame): DataFrame containing movie information
            
        Returns:
            pd.DataFrame: DataFrame with one-hot encoded genre features
        """
        # Split the genres string and create one-hot encoded columns
        genres = movies_df['genres'].str.get_dummies('|')
        return pd.concat([movies_df, genres], axis=1)

    def reduce_dimensionality(self, matrix: pd.DataFrame, n_components: int = 100) -> Tuple[np.ndarray, TruncatedSVD]:
        """
        Reduce matrix dimensionality using TruncatedSVD.
        
        Args:
            matrix (pd.DataFrame): User-movie matrix
            n_components (int): Number of components to keep
            
        Returns:
            Tuple[np.ndarray, TruncatedSVD]: Reduced matrix and fitted SVD object
        """
        self.svd = TruncatedSVD(n_components=n_components)
        reduced_matrix = self.svd.fit_transform(matrix)
        return reduced_matrix, self.svd

    def get_feature_importance(self) -> pd.Series:
        """
        Get feature importance scores after dimensionality reduction.
        
        Returns:
            pd.Series: Feature importance scores
        """
        if self.svd is None:
            raise ValueError("Must run dimensionality reduction first")
            
        return pd.Series(
            self.svd.explained_variance_ratio_,
            index=[f"component_{i}" for i in range(len(self.svd.explained_variance_ratio_))]
        )