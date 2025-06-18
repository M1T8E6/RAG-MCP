"""
Base class for MCP tools
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List

from mcp.types import Resource, TextContent, Tool


class BaseTool(ABC):
    """Base class for all MCP tools"""

    @abstractmethod
    def get_name(self) -> str:
        """Returns the tool name"""
        pass

    @abstractmethod
    def get_description(self) -> str:
        """Returns the tool description"""
        pass

    @abstractmethod
    def get_input_schema(self) -> Dict[str, Any]:
        """Returns the JSON schema for input parameters"""
        pass

    @abstractmethod
    async def execute(self, **kwargs) -> List[TextContent]:
        """Executes the tool with provided parameters and returns TextContent"""
        pass

    def to_mcp_tool(self) -> Tool:
        """Converts the tool to MCP Tool format"""
        return Tool(
            name=self.get_name(),
            description=self.get_description(),
            inputSchema=self.get_input_schema(),
        )

    def get_resources(self) -> List[Resource]:
        """Returns resources associated with this tool"""
        return []

    def handle_error(self, error: Exception, context: str = "") -> List[TextContent]:
        """Handles errors in a standardized way"""
        error_msg = f"Error in {context}: {str(error)}"
        return [TextContent(type="text", text=error_msg)]
