from langchain_core.tools import tool

from app.recommendation_engine import RecommendationEngine

ENGINE = RecommendationEngine()


@tool
def get_recommendations_for_customer(user_id: str, top_n: int = 5) -> str:
    """Return product recommendations for a customer using collaborative filtering and SVD."""
    try:
        user_based = ENGINE.get_user_similarity_recommendations(user_id, top_n=top_n)
        svd_based = ENGINE.get_svd_recommendations(user_id, top_n=top_n)
        ensemble = ENGINE.get_ensemble_recommendations(user_id, top_n=top_n)

        lines = [f"Recommendations for {user_id}:"]
        lines.append("User-based: " + (
            ", ".join(f"{item} ({score:.3f})" for item, score in user_based.items())
            if len(user_based) else "no recommendations"
        ))
        lines.append("SVD-based: " + (
            ", ".join(f"{item} ({score:.3f})" for item, score in svd_based.items())
            if len(svd_based) else "no recommendations"
        ))
        lines.append("Ensemble: " + (
            ", ".join(f"{item} ({score:.3f})" for item, score in ensemble.items())
            if len(ensemble) else "no recommendations"
        ))
        return "\n".join(lines)
    except ValueError as exc:
        return str(exc)


@tool
def get_customer_metrics(user_id: str) -> str:
    """Return customer-level recommendation metrics and rating behavior."""
    try:
        stats = ENGINE.get_user_stats(user_id)
        return (
            f"Customer {user_id}: rated {stats['items_rated']} products, "
            f"average rating {stats['avg_rating']:.2f}, "
            f"min {stats['min_rating']:.2f}, max {stats['max_rating']:.2f}, "
            f"unrated items {stats['items_not_rated']}."
        )
    except ValueError as exc:
        return str(exc)


@tool
def compare_recommendation_strategies(user_id: str, top_n: int = 5) -> str:
    """Compare user-based and SVD strategies and explain the trade-off."""
    try:
        user_based = ENGINE.get_user_similarity_recommendations(user_id, top_n=top_n)
        svd_based = ENGINE.get_svd_recommendations(user_id, top_n=top_n)
        ensemble = ENGINE.get_ensemble_recommendations(user_id, top_n=top_n)

        return (
            f"For {user_id}, user-based recommendations are: {list(user_based.index)}. "
            f"SVD recommendations are: {list(svd_based.index)}. "
            f"The ensemble combines both signals: {list(ensemble.index)}. "
            "User-based filtering is strong when similar customers exist; SVD is stronger for latent patterns and sparse data."
        )
    except ValueError as exc:
        return str(exc)


TOOLS = [get_recommendations_for_customer, get_customer_metrics, compare_recommendation_strategies]
