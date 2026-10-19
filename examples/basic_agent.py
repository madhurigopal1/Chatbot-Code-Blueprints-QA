import os

from dotenv import load_dotenv

from app.agent import run_agent


load_dotenv()


if __name__ == "__main__":
    question = (
        "I need a blueprint for an enterprise support chatbot with governance, "
        "MLOps, and human approval workflows."
    )
    print(run_agent(question))
