"""
RAG MCP Server
"""

import asyncio
import json
import logging
from typing import Any, Dict, List

from dotenv import load_dotenv
from mcp import types
from mcp.server import Server
from mcp.server.stdio import stdio_server

# Import handling for both module and direct execution
try:
    from .tools.rag_tools import RagDocsTool
except ImportError:
    # Fallback for direct execution
    import os
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from tools.rag_tools import RagDocsTool

# Load environment variables from .env file
load_dotenv()

# Logging configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


server = Server("RAG MCP Server")

# Global variable per i tools, sarà inizializzata con i parametri
AVAILABLE_TOOLS = []

def initialize_rag_tools(api_token: str, base_url: str = "https://api.ragnet-ai.com"):
    """Inizializza i tools RAG con i parametri forniti"""
    global AVAILABLE_TOOLS
    
    if not api_token:
        logger.error("RAG_API_TOKEN parameter not provided")
        raise ValueError("RAG_API_TOKEN parameter is required")

    AVAILABLE_TOOLS = [
        RagDocsTool(api_token, base_url)
    ]
    
    logger.info(f"RAG tools initialized with base URL: {base_url}")
    return AVAILABLE_TOOLS


@server.list_tools()
async def handle_list_tools() -> List[types.Tool]:
    """Elenca i tool disponibili"""
    logger.info("Listing available tools")
    
    if not AVAILABLE_TOOLS:
        return [
            types.Tool(
                name="configure_rag",
                description="Configure RAG tools with API token and base URL",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "api_token": {
                            "type": "string",
                            "description": "RAG API token for authentication"
                        },
                        "base_url": {
                            "type": "string",
                            "description": "Base URL for RAG API (optional, defaults to https://api.ragnet-ai.com)",
                            "default": "https://api.ragnet-ai.com"
                        }
                    },
                    "required": ["api_token"]
                }
            )
        ]
    
    tools = []
    for tool in AVAILABLE_TOOLS:
        tools.append(
            types.Tool(
                name=tool.get_name(),
                description=tool.get_description(),
                inputSchema=tool.get_input_schema(),
            )
        )

    return tools


@server.call_tool()
async def handle_call_tool(
    name: str, arguments: Dict[str, Any]
) -> List[types.TextContent]:
    """Gestisce le chiamate ai tool"""
    logger.info("Handling tool call: %s with arguments: %s", name, arguments)
    try:
        # Gestione tool di configurazione
        if name == "configure_rag":
            api_token = arguments.get("api_token")
            base_url = arguments.get("base_url", "https://api.ragnet-ai.com")
            
            if not api_token:
                return [
                    types.TextContent(
                        type="text",
                        text="Error: api_token parameter is required for configuration",
                    )
                ]

            try:
                initialize_rag_tools(api_token, base_url)
                return [
                    types.TextContent(
                        type="text",
                        text=f"RAG tools configured successfully with base URL: {base_url}",
                    )
                ]
            except Exception as e:
                return [
                    types.TextContent(
                        type="text",
                        text=f"Error configuring RAG tools: {str(e)}",
                    )
                ]

        # Verifica che i tools siano stati configurati
        if not AVAILABLE_TOOLS:
            return [
                types.TextContent(
                    type="text",
                    text="RAG tools not configured. Please call 'configure_rag' tool first with your API token.",
                )
            ]

        # Trova il tool corrispondente
        tool = None
        for t in AVAILABLE_TOOLS:
            if t.get_name() == name:
                tool = t
                break

        if not tool:
            return [
                types.TextContent(
                    type="text",
                    text=f"Tool '{name}' not found. Available tools: {[t.get_name() for t in AVAILABLE_TOOLS]}",
                )
            ]

        # Esegui il tool
        result = await tool.execute(**arguments)

        # Formatta il risultato
        if isinstance(result, dict):
            if result.get("success", False):
                message = result.get("message", "Operation completed successfully")
                details = []

                # Gestione specifica per risultati RAG
                if "data" in result:
                    rag_data = result["data"]
                    if isinstance(rag_data, dict):
                        # Estrai la risposta RAG
                        if "answer" in rag_data:
                            details.append(f"Risposta: {rag_data['answer']}")
                        elif "response" in rag_data:
                            details.append(f"Risposta: {rag_data['response']}")
                        
                        # Estrai informazioni sui documenti sorgente
                        if "sources" in rag_data:
                            details.append("Documenti sorgente:")
                            for i, source in enumerate(rag_data["sources"], 1):
                                if isinstance(source, dict):
                                    doc_info = f"  {i}. "
                                    if "document" in source:
                                        doc_info += f"Documento: {source['document']}"
                                    if "score" in source:
                                        doc_info += f" (Score: {source['score']:.3f})"
                                    if "content" in source:
                                        content = source['content'][:200] + "..." if len(source['content']) > 200 else source['content']
                                        doc_info += f"\n     Contenuto: {content}"
                                    details.append(doc_info)
                                else:
                                    details.append(f"  {i}. {source}")
                        
                        # Se abbiamo solo i dati grezzi, mostrali
                        if not details and rag_data:
                            details.append(f"Dati RAG: {json.dumps(rag_data, indent=2, ensure_ascii=False)}")

                if "query" in result:
                    details.append(f"Query: {result['query']}")

                # Se non ci sono dettagli specifici RAG, usa il messaggio base
                if details:
                    full_message = "\n".join(details)
                else:
                    full_message = message

                return [types.TextContent(type="text", text=full_message)]
            else:
                error_message = result.get("error", "Unknown error")
                user_message = result.get("message", "Operation failed")
                return [
                    types.TextContent(
                        type="text",
                        text=f"{user_message}\n\nError details: {error_message}",
                    )
                ]
        else:
            return [types.TextContent(type="text", text=str(result))]

    except (ValueError, TypeError, KeyError) as e:
        logger.error("Errors in tool parameters %s: %s", name, str(e), exc_info=True)
        return [
            types.TextContent(
                type="text", text=f"Errors in tool parameters '{name}': {str(e)}"
            )
        ]
    except Exception as e:
        logger.error(
            "Unexpected error occurred while executing tool %s: %s",
            name,
            str(e),
            exc_info=True,
        )
        return [
            types.TextContent(
                type="text",
                text=f"Unexpected error occurred while executing tool '{name}': {str(e)}",
            )
        ]


@server.list_resources()
async def handle_list_resources() -> List[types.Resource]:
    """Elenca le risorse disponibili"""
    logger.info("Listing available resources")
    return [
        types.Resource(
            uri="rag://docs",
            name="RAG MCP Server Documentation",
            description="Documentation for using the RAG MCP server",
            mimeType="text/plain",
        )
    ]


@server.read_resource()
async def handle_read_resource(uri: str) -> str:
    """Legge il contenuto di una risorsa"""
    logger.info("Reading resource: %s", uri)
    if uri == "rag://docs":
        return """
            # RAG MCP Server

            This MCP server provides access to RAG (Retrieval-Augmented Generation) features.

            ## Available Tools:

            ### rag_docs
            Executes a RAG query including information about source documents.
            Parameters:
            - query (string): The question to ask the RAG system

            ## Configuration:
            1. Set RAG_API_TOKEN environment variable with your API token
            2. Optionally set RAG_BASE_URL (defaults to https://api.ragnet-ai.com)
            3. Server runs in stdio mode for MCP integration

            ## Example usage:
            - Ask questions about your documents
            - Get contextual answers based on your vector database
            - Retrieve source document information when needed
        """
    else:
        raise ValueError(f"Resource not found: {uri}")


async def main():
    """Funzione principale per avviare il server"""
    try:
        logger.info("Starting RAG MCP server")
        
        # Modalità stdio per MCP
        logger.info("Starting STDIO server")
        async with stdio_server() as (read_stream, write_stream):
            await server.run(
                read_stream, write_stream, server.create_initialization_options()
            )

    except KeyboardInterrupt:
        logger.info("Server stopped by user")
    except Exception as e:
        logger.error(
            "Unexpected error occurred while starting the server: %s",
            str(e),
            exc_info=True,
        )
        raise


if __name__ == "__main__":
    asyncio.run(main())
