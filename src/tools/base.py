"""
Base class for MCP tools
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List

from mcp.types import Resource, TextContent, Tool


class BaseTool(ABC):
    """Classe base per tutti i tools MCP"""

    @abstractmethod
    def get_name(self) -> str:
        """Restituisce il nome del tool"""
        pass

    @abstractmethod
    def get_description(self) -> str:
        """Restituisce la descrizione del tool"""
        pass

    @abstractmethod
    def get_input_schema(self) -> Dict[str, Any]:
        """Restituisce lo schema JSON dei parametri di input"""
        pass

    @abstractmethod
    async def execute(self, **kwargs) -> List[TextContent]:
        """Esegue il tool con i parametri forniti e restituisce TextContent"""
        pass

    def to_mcp_tool(self) -> Tool:
        """Converte il tool nel formato MCP Tool"""
        return Tool(
            name=self.get_name(),
            description=self.get_description(),
            inputSchema=self.get_input_schema(),
        )

    def get_resources(self) -> List[Resource]:
        """Restituisce le risorse associate a questo tool"""
        return []

    def handle_error(self, error: Exception, context: str = "") -> List[TextContent]:
        """Gestisce gli errori in modo standardizzato"""
        error_msg = f"Error in {context}: {str(error)}"
        return [TextContent(type="text", text=error_msg)]
