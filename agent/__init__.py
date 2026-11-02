"""Agent package providing native Ollama tool calling, confirmation management, and dependency contracts."""

from agent.confirm import ConfirmationManager, PendingAction
from agent.deps import AgentDeps
from agent.loop import AgentResult, run_agent_loop
from agent.tools import TOOL_REGISTRY, TOOLS, ToolContext, validate_and_execute_tool

__all__ = [
    "AgentDeps",
    "AgentResult",
    "ConfirmationManager",
    "PendingAction",
    "TOOL_REGISTRY",
    "TOOLS",
    "ToolContext",
    "run_agent_loop",
    "validate_and_execute_tool",
]

