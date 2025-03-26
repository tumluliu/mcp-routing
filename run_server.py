#!/usr/bin/env python
"""Run script for MCP Routing service."""

import os
import sys
import uvicorn
from pathlib import Path
from loguru import logger

# Add the project root to the Python path
sys.path.append(str(Path(__file__).parent))

# Configure loguru for the server startup process
logs_dir = Path("logs")
logs_dir.mkdir(exist_ok=True)

# Remove default handler and set up our own handlers
logger.remove()
logger.add(
    "logs/server.log",
    rotation="10 MB",
    retention="1 week",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {name}:{function}:{line} - {message}",
    level="DEBUG",
    backtrace=True,
    diagnose=True,
)
logger.add(lambda msg: print(msg), level="INFO", format="{level} | {message}")

logger.info("Starting MCP Routing server")

try:
    # Import the app directly
    logger.info("Importing API components")
    from mcp_routing.api import app
    from mcp_routing.config import SERVICE_HOST, SERVICE_PORT, RELOAD

    logger.info(
        f"Configuration loaded: host={SERVICE_HOST}, port={SERVICE_PORT}, reload={RELOAD}"
    )
except Exception as e:
    logger.exception(f"Failed to import required modules: {str(e)}")
    sys.exit(1)

if __name__ == "__main__":
    try:
        # Create uploads directory if it doesn't exist
        uploads_dir = Path("uploads")
        uploads_dir.mkdir(exist_ok=True)
        logger.info(f"Ensuring uploads directory exists at {uploads_dir}")

        # Run the FastAPI app
        logger.info(
            f"Starting uvicorn server on {SERVICE_HOST}:{SERVICE_PORT} (reload={RELOAD})"
        )
        uvicorn.run(app, host=SERVICE_HOST, port=SERVICE_PORT, reload=RELOAD)
    except Exception as e:
        logger.exception(f"Server startup failed: {str(e)}")
        sys.exit(1)
