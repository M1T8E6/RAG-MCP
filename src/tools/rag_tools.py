"""
RAG tools for MCP server
"""

import asyncio
import logging
from typing import Any, Dict, List

import aiohttp
from mcp.server import NotificationOptions
from mcp.types import TextContent

from .base import BaseTool

logger = logging.getLogger(__name__)


class RagTool(BaseTool):
    """Tool for retrieving documents from a Vector Database"""

    def __init__(self, api_token: str, base_url: str = "https://your-service.com"):
        self.api_token = api_token
        self.base_url = base_url
        self.endpoint = f"{base_url}/api/v1/rag/docs"

        self.notification_options = NotificationOptions(
            tools_changed=True,
            resources_changed=True,
            prompts_changed=True,
        )

    def get_name(self) -> str:
        return "rag_docs"

    def get_description(self) -> str:
        return """
        It performs an intelligent query on the RAG document database.
        This tool searches company documents and provides answers based on actual content, including references to source documents. Ideal for finding specific information, details about people, processes, products or any documented content.
        """

    def get_input_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "La domanda da porre al sistema RAG",
                }
            },
            "required": ["query"],
        }

    async def execute(self, **kwargs) -> List[TextContent]:
        """Execute the RAG query with document information"""
        try:
            query = kwargs.get("query")
            if not query:
                return [
                    TextContent(type="text", text="Error: Query parameter is required")
                ]

            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_token}",
            }

            payload = {
                "query": query,
                "include_sources": True,
                "max_tokens": 500,
                "temperature": 0.1,
            }

            logger.info("Executing RAG docs query: %s", query)

            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self.endpoint,
                    headers=headers,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=30),
                ) as response:

                    if response.status == 200:
                        result = await response.json()
                        return {
                            "success": True,
                            "message": "Query executed successfully",
                            "data": result,
                            "query": query,
                        }
                    else:
                        error_text = await response.text()
                        logger.error(
                            "RAG Docs API error %d: %s", response.status, error_text
                        )
                        return [
                            TextContent(
                                type="text",
                                text=f"API error {response.status}: {error_text}",
                            ),
                            TextContent(
                                type="text",
                                text=f"Errore nella chiamata RAG Docs (status {response.status})",
                            ),
                        ]

        except asyncio.TimeoutError:
            logger.error("RAG Docs API timeout")
            return [
                TextContent(type="text", text="Request timeout"),
                TextContent(
                    type="text", text="Timeout nella chiamata al servizio RAG Docs"
                ),
            ]
        except Exception as e:
            logger.error("Unexpected error in RAG docs: %s", str(e), exc_info=True)
            return [
                TextContent(
                    type="text",
                    text=f"Errore imprevisto nella query RAG Docs: {str(e)}",
                )
            ]
