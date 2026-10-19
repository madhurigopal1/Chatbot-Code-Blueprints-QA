import os
from typing import Sequence, TypedDict

from dotenv import load_dotenv
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import START, END, StateGraph
from langgraph.prebuilt import ToolNode

from app.recommender_tools import (
    compare_recommendation_strategies,
    get_customer_metrics,
    get_recommendations_for_customer,
)

load_dotenv()


@tool
def blueprint_for_use_case(use_case: str) -> str:
    """Return a practical blueprint for a given use case."""
    use_case = use_case.lower()
    if "retail" in use_case:
        return (
            "Retail agent blueprint: 1) ingestion of product catalog and customer events, "
            "2) retrieval from a product knowledge base, 3) orchestration via a tool-calling agent, "
            "4) guardrails for pricing and policy checks, 5) human escalation for exceptions."
        )
    if "support" in use_case:
        return (
            "Support agent blueprint: 1) ticket intake and triage, 2) retrieval from knowledge base, "
            "3) policy-aware answer generation, 4) workflow actions for case updates, 5) audit logs."
        )
    if "finance" in use_case:
        return (
            "Finance agent blueprint: 1) secure data access controls, 2) retrieval of approved docs, "
            "3) deterministic calculations, 4) approval checks for actions, 5) explainability logs."
        )
    return (
        "Generic agent blueprint: 1) define goals and constraints, 2) connect to trusted tools and data, "
        "3) add governance and human review, 4) use evaluation and observability for iteration."
    )


@tool
def governance_checklist(domain: str) -> str:
    """Return a practical governance checklist for an AI domain."""
    return (
        f"Governance checklist for {domain or 'general AI'}: "
        "- data access controls and secrets management, "
        "- prompt and tool safety validation, "
        "- audit logs and observability, "
        "- human approval for high-risk actions, "
        "- periodic evaluation and drift monitoring."
    )


@tool
def mlops_blueprint(workload: str) -> str:
    """Return a sample MLOps blueprint for a GenAI workload."""
    return (
        f"MLOps blueprint for {workload}: "
        "- version models and prompts, "
        "- run evaluation pipelines, "
        "- store embeddings and retrieval metadata, "
        "- validate production changes via regression tests, "
        "- monitor quality, latency, and cost."
    )


TOOLS = [
    blueprint_for_use_case,
    governance_checklist,
    mlops_blueprint,
    get_recommendations_for_customer,
    get_customer_metrics,
    compare_recommendation_strategies,
]


class AgentState(TypedDict):
    messages: Sequence[BaseMessage]


def build_agent():
    model = ChatOpenAI(
        model=os.getenv("MODEL_NAME", "gpt-4o-mini"),
        temperature=0,
    )
    model_with_tools = model.bind_tools(TOOLS)

    def call_model(state: AgentState):
        response = model_with_tools.invoke(state["messages"])
        return {"messages": [response]}

    workflow = StateGraph(AgentState)
    workflow.add_node("agent", call_model)
    workflow.add_node("tools", ToolNode(TOOLS))
    workflow.add_edge(START, "agent")
    workflow.add_conditional_edges(
        "agent",
        lambda state: "tools" if state["messages"][-1].tool_calls else END,
        {"tools": "tools", END: END},
    )
    workflow.add_edge("tools", "agent")
    return workflow.compile()


def run_agent(prompt: str):
    app = build_agent()
    result = app.invoke({"messages": [HumanMessage(content=prompt)]})
    return result["messages"][-1].content


if __name__ == "__main__":
    demo_prompt = (
        "Recommend products for customer C003 and explain whether the user-based or SVD approach is more suitable."
    )
    print(run_agent(demo_prompt))
