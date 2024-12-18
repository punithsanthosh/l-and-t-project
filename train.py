import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import precision_score, recall_score, f1_score

# Load the datasets
movies = pd.read_csv('movies.csv')  # Movie metadata (movieId, title, genres)
ratings = pd.read_csv('ratings.csv')  # User ratings (userId, movieId, rating, timestamp)

# Create the User-Item Matrix
user_item_matrix = ratings.pivot(index='userId', columns='movieId', values='rating')

# Fill NaN with 0 for similarity calculations
user_item_matrix_filled = user_item_matrix.fillna(0)

# User-User Similarity Matrix
user_similarity = cosine_similarity(user_item_matrix_filled)
user_similarity_df = pd.DataFrame(user_similarity, index=user_item_matrix.index, columns=user_item_matrix.index)

# Item-Item Similarity Matrix
item_similarity = cosine_similarity(user_item_matrix_filled.T)
item_similarity_df = pd.DataFrame(item_similarity, index=user_item_matrix.columns, columns=user_item_matrix.columns)

def get_movie_titles(movie_ids):
    """
    Convert a list of movie IDs to their corresponding titles.

    Args:
        movie_ids (list): List of movie IDs.

    Returns:
        list: List of movie titles.
    """
    return movies[movies['movieId'].isin(movie_ids)]['title'].tolist()

def recommend_movies_user(user_id, num_recommendations=5):
    """
    Generate recommendations using User-Based Collaborative Filtering.

    Args:
        user_id (int): User ID.
        num_recommendations (int): Number of recommendations to generate.

    Returns:
        list: Top recommended movie titles.
    """
    similar_users = user_similarity_df[user_id].sort_values(ascending=False)
    weighted_ratings = user_item_matrix.mul(similar_users, axis=0).sum(axis=0)
    recommendation_scores = weighted_ratings / similar_users.sum()

    watched_movies = user_item_matrix.loc[user_id].dropna().index
    recommendations = recommendation_scores.drop(watched_movies).sort_values(ascending=False).head(num_recommendations)

    return get_movie_titles(recommendations.index.tolist())

def recommend_movies_item(user_id, num_recommendations=5):
    """
    Generate recommendations using Item-Based Collaborative Filtering.

    Args:
        user_id (int): User ID.
        num_recommendations (int): Number of recommendations to generate.

    Returns:
        list: Top recommended movie titles.
    """
    user_ratings = user_item_matrix.loc[user_id].dropna()
    weighted_ratings = item_similarity_df.loc[user_ratings.index].T.dot(user_ratings)
    recommendation_scores = weighted_ratings / item_similarity_df.loc[user_ratings.index].sum(axis=0)

    watched_movies = user_ratings.index
    recommendations = recommendation_scores.drop(watched_movies).sort_values(ascending=False).head(num_recommendations)

    return get_movie_titles(recommendations.index.tolist())

def recommend_and_evaluate(user_id, method, num_recommendations=5):
    """
    Generate recommendations and evaluate them.

    Args:
        user_id (int): User ID.
        method (str): Recommendation method ('user' or 'item').
        num_recommendations (int): Number of recommendations to generate.

    Returns:
        dict: Recommendations and evaluation metrics (Precision, Recall, F1 Score).
    """
    # Step 1: Generate Recommendations
    if method == 'user':
        recommendations = recommend_movies_user(user_id, num_recommendations)
    elif method == 'item':
        recommendations = recommend_movies_item(user_id, num_recommendations)
    else:
        raise ValueError("Invalid recommendation method.")

    print(f"Recommendations for user {user_id}: {recommendations}")

    # Step 2: Generate Ground Truth
    user_ratings = user_item_matrix.loc[user_id]
    ground_truth = user_ratings[user_ratings >= 4.0].index.tolist()

    # If no ground truth, return zero metrics
    

    print(f"Ground truth (rated >= 4.0) for user {user_id}: {ground_truth}")

    # Step 3: Map Recommended Titles to Movie IDs
    recommended_movie_ids = []
    for title in recommendations:
        movie_id = movies[movies['title'].str.strip().str.lower() == title.lower().strip()]['movieId'].tolist()
        if movie_id:
            recommended_movie_ids.extend(movie_id)

    recommended_movie_ids = list(set(recommended_movie_ids))
    print(f"Recommended movie IDs: {recommended_movie_ids}")

    # Step 4: Calculate Metrics
    true_positives = len(set(ground_truth).intersection(set(recommended_movie_ids)))
    total_recommended = len(recommended_movie_ids)
    total_relevant = len(ground_truth)

    # Precision, Recall, and F1 Score
    precision = true_positives / total_recommended if total_recommended > 0 else 0.0
    recall = true_positives / total_relevant if total_relevant > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0.0

    # Return results
    return {
        "recommendations": recommendations,
        "evaluation": {
            "Precision": precision,
            "Recall": recall,
            "F1 Score": f1
        }
    }
