"""
Data preprocessing module for Movie Recommendation System.
Handles data loading, cleaning, and preprocessing operations.
"""

import pandas as pd
import numpy as np
import json
from sklearn.preprocessing import StandardScaler

class DataPreprocessor:
    def __init__(self):
        """Initialize the DataPreprocessor."""
        self.movies = None
        self.ratings = None
        self.user_movie_matrix = None
        self.scaler = StandardScaler()

    def load_data(self, movies_path: str, ratings_path: str) -> None:
        """
        Load movie and rating data from CSV files.
        
        Args:
            movies_path (str): Path to the movies CSV file
            ratings_path (str): Path to the ratings CSV file
        """
        # Read the data files
        self.movies = pd.read_csv(movies_path)
        self.ratings = pd.read_csv(ratings_path)
        
        # Process movies dataframe to extract required columns
        self.process_movies_data()

    def process_movies_data(self) -> None:
        """Process the movies dataframe to verify required columns."""
        if self.movies is None:
            raise ValueError("Movies data must be loaded first")

        required_columns = {'movieId', 'title', 'genres'}
        missing_columns = required_columns - set(self.movies.columns)
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")

    def merge_data(self) -> pd.DataFrame:
        """
        Merge movies and ratings dataframes.
        
        Returns:
            pd.DataFrame: Merged dataframe containing movie and rating data
        """
        if self.movies is None or self.ratings is None:
            raise ValueError("Data must be loaded before merging")
        
        merged_data = pd.merge(self.ratings, self.movies, on='movieId')
        return merged_data.dropna()

    def create_user_movie_matrix(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Create user-movie rating matrix.
        
        Args:
            data (pd.DataFrame): Merged movie and rating data
            
        Returns:
            pd.DataFrame: User-movie matrix with users as rows and movies as columns
        """
        self.user_movie_matrix = data.pivot_table(
            index='userId',
            columns='title',
            values='rating'
        )
        return self.user_movie_matrix

    def handle_missing_values(self, strategy: str = 'zero') -> pd.DataFrame:
        """
        Handle missing values in the user-movie matrix.
        
        Args:
            strategy (str): Strategy for handling missing values ('zero' or 'mean')
            
        Returns:
            pd.DataFrame: Processed user-movie matrix
        """
        if self.user_movie_matrix is None:
            raise ValueError("User-movie matrix must be created first")

        if strategy == 'zero':
            return self.user_movie_matrix.fillna(0)
        elif strategy == 'mean':
            return self.user_movie_matrix.fillna(self.user_movie_matrix.mean())
        else:
            raise ValueError("Invalid strategy. Use 'zero' or 'mean'")

    def normalize_ratings(self, matrix: pd.DataFrame) -> np.ndarray:
        """
        Normalize ratings using StandardScaler.
        
        Args:
            matrix (pd.DataFrame): User-movie matrix
            
        Returns:
            np.ndarray: Normalized user-movie matrix
        """
        return self.scaler.fit_transform(matrix)