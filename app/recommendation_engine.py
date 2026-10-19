import warnings

import numpy as np
import pandas as pd
from scipy.sparse.linalg import svds
from sklearn.metrics.pairwise import cosine_similarity

warnings.filterwarnings("ignore")


def _build_sample_data():
    rows = [
        ("C001", "P001", 5),
        ("C001", "P002", 4),
        ("C001", "P003", 3),
        ("C002", "P001", 4),
        ("C002", "P004", 5),
        ("C002", "P005", 4),
        ("C003", "P002", 5),
        ("C003", "P003", 4),
        ("C003", "P006", 3),
        ("C004", "P001", 3),
        ("C004", "P004", 4),
        ("C004", "P007", 5),
        ("C005", "P003", 5),
        ("C005", "P005", 3),
        ("C005", "P008", 4),
        ("C006", "P002", 4),
        ("C006", "P006", 5),
        ("C006", "P008", 3),
    ]
    return pd.DataFrame(rows, columns=["customer_id", "product_id", "rating"])


class RecommendationEngine:
    """Retail recommendation engine using collaborative filtering and SVD."""

    def __init__(self, df=None):
        self.df = df if df is not None else _build_sample_data()
        required_columns = ["customer_id", "product_id", "rating"]
        missing = [col for col in required_columns if col not in self.df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
        self.user_item_matrix = self._create_user_item_matrix()

    def _create_user_item_matrix(self):
        return self.df.pivot_table(
            index="customer_id",
            columns="product_id",
            values="rating",
            fill_value=0,
        )

    def get_user_similarity_recommendations(self, user_id, top_n=5, num_similar_users=5):
        if user_id not in self.user_item_matrix.index:
            raise ValueError(f"User {user_id} not found in data")

        similarity = cosine_similarity(self.user_item_matrix)
        similarity_df = pd.DataFrame(
            similarity,
            index=self.user_item_matrix.index,
            columns=self.user_item_matrix.index,
        )

        similar_users = similarity_df[user_id].sort_values(ascending=False)[1 : num_similar_users + 1]

        candidate_items = pd.Series(dtype=float)
        for sim_user in similar_users.index:
            candidate_items = candidate_items.add(
                self.user_item_matrix.loc[sim_user] * similar_users[sim_user],
                fill_value=0,
            )

        already_purchased = self.user_item_matrix.loc[user_id]
        recommendations = candidate_items[already_purchased == 0]
        return recommendations.sort_values(ascending=False).head(top_n)

    def get_svd_recommendations(self, user_id, top_n=5, n_factors=3):
        if user_id not in self.user_item_matrix.index:
            raise ValueError(f"User {user_id} not found in data")

        matrix = self.user_item_matrix.values
        user_ratings_mean = np.mean(matrix, axis=1)
        matrix_demeaned = matrix - user_ratings_mean.reshape(-1, 1)
        k = min(n_factors, min(matrix_demeaned.shape) - 1)

        U, sigma, Vt = svds(matrix_demeaned, k=k)
        sigma = np.diag(sigma)
        predicted_ratings = np.dot(np.dot(U, sigma), Vt) + user_ratings_mean.reshape(-1, 1)

        preds_df = pd.DataFrame(
            predicted_ratings,
            index=self.user_item_matrix.index,
            columns=self.user_item_matrix.columns,
        )

        user_predictions = preds_df.loc[user_id]
        already_purchased = self.user_item_matrix.loc[user_id]
        recommendations = user_predictions[already_purchased == 0]
        return recommendations.sort_values(ascending=False).head(top_n)

    def get_ensemble_recommendations(self, user_id, top_n=5, weight_user_based=0.5):
        user_based = self.get_user_similarity_recommendations(user_id, top_n=top_n * 2)
        svd_based = self.get_svd_recommendations(user_id, top_n=top_n * 2)

        if len(user_based) > 0:
            user_based_norm = (user_based - user_based.min()) / (user_based.max() - user_based.min() + 1e-8)
        else:
            user_based_norm = pd.Series()

        if len(svd_based) > 0:
            svd_based_norm = (svd_based - svd_based.min()) / (svd_based.max() - svd_based.min() + 1e-8)
        else:
            svd_based_norm = pd.Series()

        ensemble = pd.Series(dtype=float)
        all_items = set(user_based_norm.index) | set(svd_based_norm.index)
        for item in all_items:
            score = 0
            if item in user_based_norm.index:
                score += weight_user_based * user_based_norm[item]
            if item in svd_based_norm.index:
                score += (1 - weight_user_based) * svd_based_norm[item]
            ensemble[item] = score

        return ensemble.sort_values(ascending=False).head(top_n)

    def get_user_stats(self, user_id):
        if user_id not in self.user_item_matrix.index:
            raise ValueError(f"User {user_id} not found in data")

        user_ratings = self.user_item_matrix.loc[user_id]
        rated_items = user_ratings[user_ratings > 0]
        return {
            "user_id": user_id,
            "items_rated": len(rated_items),
            "avg_rating": float(rated_items.mean()) if len(rated_items) > 0 else 0,
            "min_rating": float(rated_items.min()) if len(rated_items) > 0 else 0,
            "max_rating": float(rated_items.max()) if len(rated_items) > 0 else 0,
            "items_not_rated": len(user_ratings[user_ratings == 0]),
        }

    def get_all_users(self):
        return list(self.user_item_matrix.index)

    def get_all_products(self):
        return list(self.user_item_matrix.columns)
