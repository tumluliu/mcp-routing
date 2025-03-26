#!/usr/bin/env python
"""Run script for MCP Routing service."""

import os
import sys
import uvicorn
from pathlib import Path

# Add the project root to the Python path
sys.path.append(str(Path(__file__).parent))

# Import the app directly
from mcp_routing.api import app
from mcp_routing.config import SERVICE_HOST, SERVICE_PORT, RELOAD

if __name__ == "__main__":
    # Create uploads directory if it doesn't exist
    uploads_dir = Path("uploads")
    uploads_dir.mkdir(exist_ok=True)

    # Run the FastAPI app
    uvicorn.run(app, host=SERVICE_HOST, port=SERVICE_PORT, reload=RELOAD)
