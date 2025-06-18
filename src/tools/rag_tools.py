"""
RAG tools for MCP server
"""
import asyncio
import json
import logging
from typing import Any, Dict

import aiohttp

from .base import BaseTool

logger = logging.getLogger(__name__)



class RagDocsTool(BaseTool):
    """Tool per eseguire query RAG con informazioni sui documenti"""
    
    def __init__(self, api_token: str, base_url: str = "https://api.ragnet-ai.com"):
        self.api_token = api_token
        self.base_url = base_url
        self.endpoint = f"{base_url}/api/v1/rag/docs"
    
    def get_name(self) -> str:
        return "rag_docs"
    
    def get_description(self) -> str:
        return """Esegue una query intelligente sul database di documenti RAG. 
        Questo tool cerca tra i documenti dell'azienda e fornisce risposte basate sui contenuti effettivi, 
        includendo riferimenti ai documenti sorgente. Ideale per trovare informazioni specifiche, 
        dettagli su persone, processi, prodotti o qualsiasi contenuto documentato."""
    
    def get_input_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "La domanda da porre al sistema RAG"
                }
            },
            "required": ["query"]
        }
    
    async def execute(self, **kwargs) -> Dict[str, Any]:
        """Esegue la query RAG con informazioni sui documenti"""
        try:
            query = kwargs.get("query")
            if not query:
                return {
                    "success": False,
                    "error": "Query parameter is required",
                    "message": "Devi fornire una query per il sistema RAG"
                }
            
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_token}"
            }
            
            payload = {
                "query": query,
                "include_sources": True,
                "max_tokens": 500,
                "temperature": 0.1
            }
            
            logger.info(f"Executing RAG docs query: {query}")
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self.endpoint,
                    headers=headers,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=30)
                ) as response:
                    
                    if response.status == 200:
                        result = await response.json()
                        return {
                            "success": True,
                            "message": "Query RAG con documenti eseguita con successo",
                            "data": result,
                            "query": query
                        }
                    else:
                        error_text = await response.text()
                        logger.error(f"RAG Docs API error {response.status}: {error_text}")
                        return {
                            "success": False,
                            "error": f"API error {response.status}: {error_text}",
                            "message": f"Errore nella chiamata RAG Docs (status {response.status})"
                        }
                        
        except asyncio.TimeoutError:
            logger.error("RAG Docs API timeout")
            return {
                "success": False,
                "error": "Request timeout",
                "message": "Timeout nella chiamata al servizio RAG Docs"
            }
        except Exception as e:
            logger.error(f"Unexpected error in RAG docs: {str(e)}", exc_info=True)
            return {
                "success": False,
                "error": str(e),
                "message": f"Errore imprevisto nella query RAG Docs: {str(e)}"
            }
