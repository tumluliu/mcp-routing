"""Configuration settings for the MCP Routing service."""

import os
from pathlib import Path
from dotenv import load_dotenv
from typing import Dict, Any, Optional, Tuple

# Load environment variables from .env file
load_dotenv()

# Base paths
ROOT_DIR = Path(__file__).parent.parent
DATA_DIR = ROOT_DIR / "data"
MODELS_DIR = ROOT_DIR / "models"

# Ensure directories exist
DATA_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)

# DeepSeek configuration
DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
DEEPSEEK_ENDPOINT = os.getenv("DEEPSEEK_ENDPOINT", "http://localhost:8000/v1")
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")

# Routing engine configuration
ROUTING_ENGINE = os.getenv("ROUTING_ENGINE", "osrm")
OSRM_ENDPOINT = os.getenv("OSRM_ENDPOINT", "http://localhost:5000")
ORS_ENDPOINT = os.getenv("ORS_ENDPOINT", "http://localhost:8080/ors")
ORS_API_KEY = os.getenv("ORS_API_KEY", "")

# Place search configuration
OVERPASS_API_ENDPOINT = os.getenv(
    "OVERPASS_API_ENDPOINT", "https://overpass-api.de/api/interpreter"
)

# Munich bounding box (approx)
MUNICH_BBOX: Dict[str, float] = {
    "min_lat": 48.0,
    "max_lat": 48.3,
    "min_lon": 11.3,
    "max_lon": 11.8,
}

# Munich city center coordinates (Marienplatz)
MUNICH_CENTER: Tuple[float, float] = (48.1371, 11.5754)

# Extended Munich region for geocoding fallbacks
MUNICH_EXTENDED_BBOX: Dict[str, float] = {
    "min_lat": 47.8,
    "max_lat": 48.5,
    "min_lon": 11.2,
    "max_lon": 11.9,
}

# Default search radiuses for different transport modes (in meters)
DEFAULT_SEARCH_RADIUS: Dict[str, int] = {
    "driving": 5000,
    "walking": 2000,
    "cycling": 3000,
    "default": 3000,
}

# Service configuration
SERVICE_HOST = os.getenv("SERVICE_HOST", "0.0.0.0")
SERVICE_PORT = int(os.getenv("SERVICE_PORT", "8000"))
RELOAD = os.getenv("RELOAD", "False").lower() in ["true", "1", "yes"]
