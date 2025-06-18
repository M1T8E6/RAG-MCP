# RAG MCP Server

A Model Context Protocol (MCP) server that provides RAG (Retrieval-Augmented Generation) capabilities for document querying. This server enables AI assistants to perform intelligent queries on your document database and retrieve contextual answers with source information.

## Features

- 🔍 **Intelligent Document Querying**: Perform semantic search on your document collection
- 📚 **Source Attribution**: Get references to source documents with relevance scores
- 🔒 **Secure Authentication**: Token-based authentication for API access
- ⚡ **Async Performance**: Built with async/await for high-performance operations
- 🛠️ **MCP Integration**: Seamless integration with MCP-compatible AI assistants
- 🔧 **Configurable**: Flexible configuration options for different RAG backends

## Installation

### Prerequisites

- Python 3.10 or higher
- pip or uv package manager

### Using pip

```bash
# Clone the repository
git clone https://github.com/yourusername/rag-mcp.git
cd rag-mcp-server

# Install dependencies
pip install -e .
```

### Using uv (recommended)

```bash
# Clone the repository
git clone https://github.com/yourusername/rag-mcp.git
cd rag-mcp

# Install with uv
uv sync
```

## Configuration

### Environment Variables

Create a `.env` file in the project root:

```bash
# RAG API Configuration
RAG_API_TOKEN=your_api_token_here
RAG_BASE_URL=https://your-rag-service.com/api-path

# Optional: Logging level
LOG_LEVEL=INFO
```

### Configuration Options

- `RAG_API_TOKEN`: Your API token for RAG service authentication (required)
- `RAG_BASE_URL`: Base URL for your RAG API service (optional, defaults to placeholder)

## Usage

### Running the Server

#### Direct execution
```bash
python src/server.py
```

#### Using the Makefile
```bash
make run
```

### MCP Integration

Add this server to your MCP client configuration:

```json
{
  "mcpServers": {
    "rag-server": {
      "command": "python",
      "args": ["/path/to/rag-mcp-server/src/server.py"],
      "env": {
        "RAG_API_TOKEN": "your_api_token_here",
        "RAG_BASE_URL": "https://your-rag-service.com/api-path"
      }
    }
  }
}
```

## Available Tools

### `configure_rag`
Configure the RAG tools with your API credentials.

**Parameters:**
- `api_token` (string, required): Your RAG API token
- `base_url` (string, optional): Base URL for the RAG service

**Example:**
```json
{
  "api_token": "your_api_token_here",
  "base_url": "https://your-rag-service.com/api-path"
}
```

### `rag_docs`
Perform intelligent queries on your document database with source attribution.

**Parameters:**
- `query` (string, required): The question to ask the RAG system

**Example:**
```json
{
  "query": "What are the main features of our product?"
}
```

**Response includes:**
- Generated answer based on your documents
- Source document references with relevance scores
- Content snippets from relevant documents

## API Integration

The server expects your RAG API to follow this interface:

### Request Format
```json
{
  "query": "Your question here",
}
```

### Response Format
```json
{
  "answer": "Generated response",
  "sources": [
    {
      "document": "document_name.pdf",
      "score": 0.95,
      "content": "Relevant text snippet..."
    }
  ]
}
```

## Development

### Setting up Development Environment

```bash
# Install development dependencies
pip install -e ".[dev]"

# Or with uv
uv sync --group dev
```

### Code Quality

The project uses several tools for code quality:

- **Black**: Code formatting
- **Ruff**: Linting and import sorting
- **MyPy**: Type checking
- **pytest**: Testing

Run quality checks:
```bash
# Format code
black src/

# Lint code
ruff check src/

# Type checking
mypy src/

# Run tests
pytest
```

### Project Structure

```
├── src/
│   ├── __init__.py
│   ├── server.py              # Main MCP server
│   └── tools/
│       ├── __init__.py
│       ├── base.py            # Base tool class
│       └── rag_tools.py       # RAG tool implementation
├── pyproject.toml             # Project configuration
├── Makefile                   # Build commands
└── README.md                  # This file
```

## Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Make your changes
4. Run tests and quality checks
5. Commit your changes: `git commit -am 'Add feature'`
6. Push to the branch: `git push origin feature-name`
7. Submit a pull request

### Code Style

- Follow PEP 8 guidelines
- Use type hints
- Write docstrings for all public functions
- Keep functions focused and small
- Add tests for new features

## Troubleshooting

### Common Issues

**"RAG tools not configured" error**
- Make sure to call the `configure_rag` tool first with your API token

**Connection timeout errors**
- Check that your `RAG_BASE_URL` is correct and accessible
- Verify your network connection
- Ensure the RAG service is running

**Authentication failures**
- Verify your `RAG_API_TOKEN` is valid
- Check that the token has appropriate permissions

### Debug Mode

Enable debug logging by setting the environment variable:
```bash
export LOG_LEVEL=DEBUG
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Support

- 📧 **Email**: andreoni.matteo@proton.me

## Acknowledgments

- Built with [Model Context Protocol](https://github.com/modelcontextprotocol/python-sdk) 
