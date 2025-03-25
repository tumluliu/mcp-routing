# MCP Routing Service

A Model Context Protocol (MCP) service that uses DeepSeek language model and OSM-based routing engines to provide natural language routing capabilities.

## Features

- Accept natural language queries for routes
- Use DeepSeek to parse queries into structured parameters
- Support for multiple routing engines (OSRM, OpenRouteService)
- Generate human-friendly navigation instructions
- Visualize routes on interactive maps
- Web UI for easy interaction
- Command-line interface for direct usage

## Architecture

The service consists of these main components:

1. **LLM Interface**: Communicates with DeepSeek to parse natural language queries and generate navigation instructions
2. **Routing Engine**: Interfaces with OSM-based routing services like OSRM or OpenRouteService
3. **Visualization**: Creates interactive maps of routes using Folium
4. **API**: Exposes a FastAPI-based web service with a simple UI
5. **CLI**: Provides command-line access to all functionality

## Prerequisites

- Python 3.11+
- DeepSeek running locally (or accessible via API)
- OSRM or OpenRouteService running locally with Munich map data

## Installation

### Using Poetry

```bash
# Clone the repository
git clone https://github.com/yourusername/mcp-routing.git
cd mcp-routing

# Install dependencies
poetry install
```

### Using pip

```bash
# Clone the repository
git clone https://github.com/yourusername/mcp-routing.git
cd mcp-routing

# Install dependencies
pip install -e .
```

## Setting Up Local Routing Engines

### OSRM

1. Download OSM data for Munich:
```bash
mkdir -p data/osrm
wget https://download.geofabrik.de/europe/germany/bayern/oberbayern-latest.osm.pbf -O data/osrm/munich.osm.pbf
```

2. Run OSRM using Docker:
```bash
docker run -t -v "${PWD}/data/osrm:/data" osrm/osrm-backend osrm-extract -p /opt/car.lua /data/munich.osm.pbf
docker run -t -v "${PWD}/data/osrm:/data" osrm/osrm-backend osrm-partition /data/munich.osrm
docker run -t -v "${PWD}/data/osrm:/data" osrm/osrm-backend osrm-customize /data/munich.osrm
docker run -t -i -p 5000:5000 -v "${PWD}/data/osrm:/data" osrm/osrm-backend osrm-routed --algorithm mld /data/munich.osrm
```

### OpenRouteService

1. Download OSM data for Munich:
```bash
mkdir -p data/ors
wget https://download.geofabrik.de/europe/germany/bayern/oberbayern-latest.osm.pbf -O data/ors/munich.osm.pbf
```

2. Run OpenRouteService using Docker:
```bash
docker run -t -i -p 8080:8080 -v "${PWD}/data/ors:/data" openrouteservice/openrouteservice
```

### Setting Up DeepSeek Locally

You can run DeepSeek locally using various methods. One approach is to use the [deepseek-coder](https://github.com/deepseek-ai/deepseek-coder) repository with vLLM:

```bash
# Install vLLM
pip install vllm

# Run the model
python -m vllm.entrypoints.openai.api_server \
    --model deepseek-ai/deepseek-coder-33b-instruct \
    --host 0.0.0.0 \
    --port 8000
```

## Configuration

Create a `.env` file in the project root with these settings:

```bash
# DeepSeek configuration
DEEPSEEK_MODEL=deepseek-ai/deepseek-coder-33b-instruct
DEEPSEEK_ENDPOINT=http://localhost:8000/v1
DEEPSEEK_API_KEY=

# Routing engine configuration
ROUTING_ENGINE=osrm  # or openrouteservice
OSRM_ENDPOINT=http://localhost:5000
ORS_ENDPOINT=http://localhost:8080/ors
ORS_API_KEY=

# Service configuration
SERVICE_HOST=0.0.0.0
SERVICE_PORT=8000
RELOAD=False
```

## Usage

### Web UI

Start the web server:

```bash
python -m mcp_routing server
```

Then open http://localhost:8000 in your browser.

### Command Line

Process a routing query:

```bash
python -m mcp_routing route "How do I get from Marienplatz to the English Garden?"
```

With additional options:

```bash
python -m mcp_routing route "Find a route from Hauptbahnhof to Allianz Arena avoiding highways" \
  --engine openrouteservice \
  --display-map
```

## API

The service exposes a REST API at:

- `GET /`: Web UI
- `POST /route`: Process a routing query
- `GET /map/{map_id}`: Get a generated map

## Development

To set up a development environment:

```bash
poetry install --with dev
```

Run tests:

```bash
poetry run pytest
```

## License

[MIT License](LICENSE)
