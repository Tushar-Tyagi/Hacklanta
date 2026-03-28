"""
Hacklanta Agents - AI Agent framework with dual-processing capabilities.

Exports:
    BaseAgent: Base agent class with local and API processing modes
    AgentResponse: Standardized response object
    ProcessingMode: Enum for processing mode selection
    Custom Exceptions: AgentError, LocalModelError, APIFallbackError
"""

from .base_agent import (
    BaseAgent,
    AgentResponse,
    ProcessingMode,
    AgentError,
    LocalModelError,
    APIFallbackError,
)

__all__ = [
    "BaseAgent",
    "AgentResponse",
    "ProcessingMode",
    "AgentError",
    "LocalModelError",
    "APIFallbackError",
]

__version__ = "1.0.0"
