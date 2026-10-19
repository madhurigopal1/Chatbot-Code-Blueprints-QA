from __future__ import annotations

from typing import Any

from langchain_core.tools import tool


@tool
def blueprint_summary(topic: str) -> str:
    """Return a concise architecture summary for common AI blueprint categories."""
    topics = {
        "chatbot": "Use a retrieval-augmented architecture with a router, tool calls, guardrails, and an agent loop.",
        "governance": "Add role-based identity, audit trails, human approval, policy checks, and evaluation pipelines.",
        "mlops": "Version prompts, models, and datasets; automate evaluations; monitor drift, latency, and cost.",
        "agentic": "Keep the system stateful, tool-driven, and observable; separate planning, execution, and review.",
    }
    return topics.get(topic.lower(), "Use a modular architecture with retrieval, orchestration, and governance layers.")


@tool
def architecture_checklist(domain: str) -> str:
    """Return a practical architecture checklist for a domain."""
    return (
        f"Checklist for {domain}: "
        "- clear ownership, "
        "- data access boundaries, "
        "- secure secrets management, "
        "- tracing and evaluation, "
        "- exception handling and escalation."
    )


@tool
def recommended_agent_pattern(domain: str) -> str:
    """Return a recommended agent pattern."""
    if "support" in domain.lower():
        return "Recommended pattern: Retrieval + tool orchestration + human escalation + policy checks."
    if "retail" in domain.lower():
        return "Recommended pattern: Planner + product search + recommendation tools + observability + guardrails."
    return "Recommended pattern: Reasoning agent + retrieval + tool use + feedback loop + audit logs."


def get_tools() -> list[Any]:
    return [blueprint_summary, architecture_checklist, recommended_agent_pattern]
