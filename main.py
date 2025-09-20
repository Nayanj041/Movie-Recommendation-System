"""
Main script for the Movie Recommendation System.
"""

import pandas as pd
import numpy as np
from src.preprocessing import DataPreprocessor
from src.feature_engineering import FeatureEngineering
from src.hybrid_recommender import HybridRecommender
from src.enhanced_evaluation import EnhancedEvaluator

def analyze_user_recommendations(user_id, recommender, processed_matrix, movies_df, evaluator):
    """Analyze recommendations for a specific user with enhanced metrics."""
    print(f"\n=== Recommendations for User {user_id} ===")
    
    # Get user's current ratings
    user_ratings = processed_matrix.loc[user_id].dropna()
    print(f"User has rated {len(user_ratings)} movies")
    
    # Show top rated movies by user
    print("\nTop 3 Movies Rated by User:")
    top_rated = user_ratings.sort_values(ascending=False)[:3]
    for movie, rating in top_rated.items():
        print(f"- {movie}: {rating:.1f}/5.0")
    
    # Get recommendations
    recommendations = recommender.recommend_movies(user_id, top_n=5)
    print("\nTop 5 Recommended Movies:")
    for movie, score in recommendations:
        movie_genres = movies_df[movies_df['title'] == movie]['genres'].iloc[0]
        print(f"- {movie} (Score: {score:.2f}) - Genres: {movie_genres}")
    
    # Calculate accuracy metrics
    if len(user_ratings) > 0:
        # Split into train/test
        train_size = int(0.8 * len(user_ratings))
        train_ratings = user_ratings[:train_size]
        test_ratings = user_ratings[train_size:]
        
        # Generate predictions for test set
        true_ratings = {}
        pred_ratings = {}
        
        for movie in test_ratings.index:
            true_ratings[movie] = test_ratings[movie]
            pred_ratings[movie] = recommender.predict_rating(user_id, movie)
        
        # Calculate multiple metrics
        rmse = evaluator.calculate_rmse(
            np.array(list(true_ratings.values())),
            np.array(list(pred_ratings.values()))
        )
        
        # Calculate NDCG
        ndcg = evaluator.ndcg_at_k(true_ratings, pred_ratings, k=5)
        
        # Get highly rated movies (ratings >= 4.0) for precision/recall
        actual_liked = [m for m, r in true_ratings.items() if r >= 4.0]
        recommended = [m for m, _ in recommendations]
        
        precision = evaluator.precision_at_k(actual_liked, recommended, k=5)
        recall = evaluator.recall_at_k(actual_liked, recommended, k=5)
        
        print(f"\nPrediction Metrics:")
        print(f"- RMSE: {rmse:.2f}")
        print(f"- NDCG@5: {ndcg:.2f}")
        print(f"- Precision@5: {precision:.2f}")
        print(f"- Recall@5: {recall:.2f}")

def analyze_movie_popularity(movie_title, processed_matrix):
    """Analyze ratings and popularity for a specific movie."""
    print(f"\n=== Analysis for Movie: {movie_title} ===")
    
    movie_ratings = processed_matrix[movie_title].dropna()
    
    print(f"Statistics:")
    print(f"- Number of ratings: {len(movie_ratings)}")
    print(f"- Average rating: {movie_ratings.mean():.2f}/5.0")
    print(f"- Rating distribution:")
    
    # Show rating distribution
    bins = np.arange(0, 5.5, 0.5)
    hist = np.histogram(movie_ratings, bins=bins)[0]
    for i, count in enumerate(hist):
        stars = f"{bins[i]:.1f}-{bins[i+1]:.1f}"
        print(f"  {stars} stars: {count:3d} ratings")

def main():
    # Initialize components
    preprocessor = DataPreprocessor()
    feature_engineer = FeatureEngineering()
    recommender = HybridRecommender(collaborative_weight=0.7)
    evaluator = EnhancedEvaluator()

    print("Loading and preprocessing data...")
    try:
        preprocessor.load_data(
            movies_path='data/movies.csv',
            ratings_path='data/ratings.csv'
        )
    except FileNotFoundError:
        print("Error: Data files not found. Please ensure movies.csv and ratings.csv exist in the data directory.")
        return

    # Process data
    merged_data = preprocessor.merge_data()
    user_movie_matrix = preprocessor.create_user_movie_matrix(merged_data)
    processed_matrix = preprocessor.handle_missing_values(strategy='zero')
    normalized_matrix = preprocessor.normalize_ratings(processed_matrix)
    user_normalized = feature_engineer.normalize_per_user(processed_matrix)
    
    # Prepare movies data for content-based filtering
    movies_data = merged_data[['movieId', 'title', 'genres']].drop_duplicates()
    movies_data.set_index('movieId', inplace=True)
    
    # Fit recommendation model
    print("Training hybrid recommendation model...")
    recommender.fit(user_normalized, movies_data)
    
    print("=== Enhanced Movie Recommendation System Analysis ===")
    
    # Analyze recommendations for multiple users with different rating patterns
    test_users = [
        1,  # Active user with many ratings
        10, # Average user
        50  # New user with few ratings
    ]
    
    print("\n=== User-based Analysis ===")
    for user_id in test_users:
        analyze_user_recommendations(user_id, recommender, processed_matrix, merged_data, evaluator)
    
    print("\n=== Content-based Analysis ===")
    # Analyze movies from different genres
    unique_genres = set()
    for genres in merged_data['genres'].str.split('|'):
        unique_genres.update(genres)
    print(f"Total unique genres: {len(unique_genres)}")
    
    # Sample movies from different genres
    genre_samples = {}
    for genre in unique_genres:
        genre_movies = merged_data[merged_data['genres'].str.contains(genre)]['title']
        if not genre_movies.empty:
            genre_samples[genre] = genre_movies.iloc[0]
    
    print("\nSample movies by genre:")
    for genre, movie in genre_samples.items():
        analyze_movie_popularity(movie, processed_matrix)
        
    # Calculate genre diversity in recommendations
    print("\n=== Recommendation Diversity Analysis ===")
    recommended_items = []
    for user_id in test_users:
        recs = recommender.recommend_movies(user_id, top_n=10)
        for movie, _ in recs:
            movie_genres = merged_data[merged_data['title'] == movie]['genres'].iloc[0].split('|')
            recommended_items.append((movie, movie_genres))
    
    diversity_score = evaluator.diversity(recommended_items, k=10)
    print(f"Recommendation diversity score: {diversity_score:.2f}")
    
    # Overall system performance
    print("\n=== Overall System Performance ===")
    all_predictions = []
    all_actuals = []
    
    # Sample 10 random users for overall evaluation
    for user_id in np.random.choice(processed_matrix.index, 10):
        user_ratings = processed_matrix.loc[user_id].dropna()
        if len(user_ratings) > 0:
            train_size = int(0.8 * len(user_ratings))
            test_ratings = user_ratings[train_size:]
            
            for movie in test_ratings.index:
                similar_users = recommender.get_similar_users(user_id)
                pred_rating = sum(similar_users[u] * processed_matrix.loc[u, movie] 
                                for u in similar_users.index 
                                if not pd.isna(processed_matrix.loc[u, movie]))
                all_predictions.append(pred_rating)
                all_actuals.append(test_ratings[movie])
    
    overall_rmse = evaluator.calculate_rmse(
        np.array(all_actuals),
        np.array(all_predictions)
    )
    
    print(f"Overall RMSE: {overall_rmse:.2f}")
    print(f"Number of users: {len(processed_matrix.index)}")
    print(f"Number of movies: {len(processed_matrix.columns)}")
    print(f"Total ratings: {len(merged_data)}")

if __name__ == "__main__":
    main()