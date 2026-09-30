"""Natural-language requests, validated plans, and evidence-backed reports."""

from .loop import AgentReport, run_agent
from .parser import ParseResult, parse_request
from .request import EmulationRequest

__all__ = ["AgentReport", "EmulationRequest", "ParseResult", "parse_request", "run_agent"]
