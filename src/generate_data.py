"""
Script to generate synthetic movie recommendation data.
This script creates sample movies.csv and ratings.csv files for testing purposes.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def generate_movies(num_movies=1000):
    """
    Generate synthetic movie data.
    
    Args:
        num_movies (int): Number of movies to generate
        
    Returns:
        pd.DataFrame: Generated movie data
    """
    # Sample movie genres
    genres = [
        "Action", "Adventure", "Animation", "Comedy", "Crime",
        "Drama", "Fantasy", "Romance", "Sci-Fi", "Thriller"
    ]
    
    # Generate movie data
    movies_data = []
    for movie_id in range(1, num_movies + 1):
        # Generate 1-3 random genres for each movie
        num_genres = np.random.randint(1, 4)
        movie_genres = '|'.join(np.random.choice(genres, num_genres, replace=False))
        
        # Generate a movie title with year
        year = np.random.randint(1990, 2024)
        title = f"Movie {movie_id} ({year})"
        
        movies_data.append({
            'movieId': movie_id,
            'title': title,
            'genres': movie_genres
        })
    
    return pd.DataFrame(movies_data)

def generate_ratings(num_users=500, num_movies=1000, ratings_per_user=200):
    """
    Generate synthetic rating data.
    
    Args:
        num_users (int): Number of users
        num_movies (int): Number of movies
        ratings_per_user (int): Number of ratings per user
        
    Returns:
        pd.DataFrame: Generated rating data
    """
    ratings_data = []
    base_timestamp = int(datetime(2020, 1, 1).timestamp())
    
    for user_id in range(1, num_users + 1):
        # Randomly select movies to rate
        movies_to_rate = np.random.choice(
            range(1, num_movies + 1), 
            size=ratings_per_user, 
            replace=False
        )
        
        for movie_id in movies_to_rate:
            # Generate rating (0.5-5.0 with 0.5 step)
            rating = np.random.choice(np.arange(0.5, 5.5, 0.5))
            # Generate timestamp
            timestamp = base_timestamp + len(ratings_data) * 3600  # 1 hour between ratings
            
            ratings_data.append({
                "userId": user_id,
                "movieId": movie_id,
                "rating": rating,
                "timestamp": timestamp
            })
    
    return pd.DataFrame(ratings_data)

def save_datasets(movies_path="data/movies.csv", ratings_path="data/ratings.csv"):
    """
    Generate and save synthetic datasets.
    
    Args:
        movies_path (str): Path to save movies dataset
        ratings_path (str): Path to save ratings dataset
    """
    # Generate datasets with larger numbers
    movies_df = generate_movies(num_movies=1000)
    ratings_df = generate_ratings(num_users=500, num_movies=1000, ratings_per_user=200)
    
    # Save to CSV
    movies_df.to_csv(movies_path, index=False)
    ratings_df.to_csv(ratings_path, index=False)
    
    print(f"Generated {len(movies_df)} movies")
    print(f"Generated {len(ratings_df)} ratings from {ratings_df['userId'].nunique()} users")
    print(f"Files saved to {movies_path} and {ratings_path}")
    
    # Print some statistics
    print("\nRating Statistics:")
    print(ratings_df['rating'].describe())
    
    print("\nSample of movies:")
    print(movies_df.head())

if __name__ == "__main__":
    save_datasets()