"""
Script to convert TMDB dataset format to MovieLens format.
"""

import pandas as pd
import json
import ast

def convert_tmdb_to_movielens():
    """Convert TMDB format movies and ratings to MovieLens format."""
    
    try:
        # Read the original TMDB format files with more careful parsing
        movies_tmdb = pd.read_csv('data/movies.csv', 
                                encoding='utf-8',
                                escapechar='\\',
                                quoting=1)  # QUOTE_ALL
        ratings = pd.read_csv('data/ratings.csv')
        
        # Process movies data
        movies_processed = pd.DataFrame({
            'movieId': movies_tmdb['id'],
            'title': movies_tmdb['title'],
            'genres': movies_tmdb['genres'].apply(lambda x: process_genres(x))
        })
        
        # Save processed files
        movies_processed.to_csv('data/movies_processed.csv', index=False)
        print("Processed movies saved to data/movies_processed.csv")
        print(f"Number of movies: {len(movies_processed)}")
        print("\nSample of processed movies:")
        print(movies_processed.head())
        
    except Exception as e:
        print(f"Error processing files: {str(e)}")
        # Try alternative reading method
        try:
            print("\nTrying alternative reading method...")
            # Read with different parameters
            movies_tmdb = pd.read_csv('data/movies.csv', 
                                    encoding='utf-8',
                                    on_bad_lines='skip',
                                    escapechar='\\')
            
            # Process movies data
            movies_processed = pd.DataFrame({
                'movieId': movies_tmdb['id'],
                'title': movies_tmdb['title'],
                'genres': movies_tmdb['genres'].apply(lambda x: process_genres(x))
            })
            
            # Save processed files
            movies_processed.to_csv('data/movies_processed.csv', index=False)
            print("Processed movies saved to data/movies_processed.csv")
            print(f"Number of movies: {len(movies_processed)}")
            print("\nSample of processed movies:")
            print(movies_processed.head())
            
        except Exception as e2:
            print(f"Alternative method also failed: {str(e2)}")

def process_genres(genres_str):
    """Convert TMDB genre format to pipe-separated string."""
    try:
        # Convert string representation of list to actual list
        genres_list = ast.literal_eval(genres_str)
        # Extract genre names and join with pipe
        return '|'.join([genre['name'] for genre in genres_list])
    except (ValueError, SyntaxError, TypeError):
        return ''

if __name__ == "__main__":
    convert_tmdb_to_movielens()