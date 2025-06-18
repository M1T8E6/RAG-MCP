"""
Base class for MCP tools
"""
from abc import ABC, abstractmethod
from typing import Any, Dict


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
    async def execute(self, **kwargs) -> Dict[str, Any]:
        """Esegue il tool con i parametri forniti"""
        pass