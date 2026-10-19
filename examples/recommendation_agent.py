import os

from dotenv import load_dotenv

from app.agent import run_agent


load_dotenv()


if __name__ == "__main__":
    prompt = (
        "Recommend products for customer C003, explain the recommendation strategy, and "
        "summarize why the ensemble method is useful for retail personalization."
    )
    print(run_agent(prompt))
