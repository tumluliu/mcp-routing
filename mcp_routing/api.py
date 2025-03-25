"""MCP Routing API using FastAPI."""

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any
import os

from .llm import DeepSeekLLM
from .routing import get_routing_engine
from .visualization import create_route_map
from .config import SERVICE_HOST, SERVICE_PORT, RELOAD

app = FastAPI(
    title="MCP Routing Service",
    description="Natural language routing service using DeepSeek and OSM-based routing engines",
    version="0.1.0",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize services
llm = DeepSeekLLM()
routing_engine = get_routing_engine()


class RoutingQuery(BaseModel):
    """Model for routing query requests."""

    query: str


class RoutingResult(BaseModel):
    """Model for routing query results."""

    query: str
    parsed_params: Dict[str, Any]
    route_data: Dict[str, Any]
    instructions: List[str]
    map_url: str


@app.get("/", response_class=HTMLResponse)
async def root():
    """Root endpoint that provides a simple UI."""
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>MCP Routing Service</title>
        <style>
            body { font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; }
            h1 { color: #3366cc; }
            input[type="text"] { width: 100%; padding: 10px; margin: 10px 0; }
            button { padding: 10px 20px; background-color: #3366cc; color: white; border: none; cursor: pointer; }
            #result { margin-top: 20px; }
            #map { height: 500px; width: 100%; border: 1px solid #ccc; margin-top: 20px; }
            #instructions { margin-top: 20px; }
        </style>
    </head>
    <body>
        <h1>MCP Routing Service</h1>
        <p>Enter your routing query in natural language:</p>
        
        <input type="text" id="query" placeholder="e.g., How do I get from Marienplatz to the English Garden?">
        <button onclick="submitQuery()">Find Route</button>
        
        <div id="result"></div>
        <div id="map"></div>
        <div id="instructions"></div>
        
        <script>
            function submitQuery() {
                const query = document.getElementById('query').value;
                if (!query) {
                    alert('Please enter a query');
                    return;
                }
                
                document.getElementById('result').innerHTML = 'Processing request...';
                
                fetch('/route', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ query: query })
                })
                .then(response => response.json())
                .then(data => {
                    // Display route details
                    const distance = (data.route_data.distance / 1000).toFixed(1);
                    const duration = Math.round(data.route_data.duration / 60);
                    
                    document.getElementById('result').innerHTML = `
                        <h3>Route Found</h3>
                        <p><strong>Distance:</strong> ${distance} km</p>
                        <p><strong>Duration:</strong> ${duration} min</p>
                        <p><a href="${data.map_url}" target="_blank">Open map in new window</a></p>
                    `;
                    
                    // Display instructions
                    const instructionsDiv = document.getElementById('instructions');
                    instructionsDiv.innerHTML = '<h3>Navigation Instructions</h3><ol>';
                    data.instructions.forEach(instruction => {
                        instructionsDiv.innerHTML += `<li>${instruction}</li>`;
                    });
                    instructionsDiv.innerHTML += '</ol>';
                    
                    // Load map in iframe
                    document.getElementById('map').innerHTML = `<iframe src="${data.map_url}" width="100%" height="500px" frameborder="0"></iframe>`;
                })
                .catch(error => {
                    document.getElementById('result').innerHTML = `<p>Error: ${error.message}</p>`;
                });
            }
        </script>
    </body>
    </html>
    """


@app.post("/route", response_model=RoutingResult)
async def route(query_data: RoutingQuery):
    """Process a natural language routing query.

    Args:
        query_data: The routing query

    Returns:
        RoutingResult: The routing result
    """
    try:
        # Parse the natural language query
        parsed_params = llm.parse_routing_query(query_data.query)

        # Check for required parameters
        if "origin" not in parsed_params or "destination" not in parsed_params:
            raise HTTPException(
                status_code=400,
                detail="Could not extract origin and destination from query",
            )

        # Get routing data
        route_data = routing_engine.route(
            origin=parsed_params["origin"],
            destination=parsed_params["destination"],
            mode=parsed_params.get("mode", "driving"),
            waypoints=parsed_params.get("waypoints"),
            avoid=parsed_params.get("avoid"),
        )

        # Generate navigation instructions
        instructions = llm.generate_navigation_instructions(route_data)

        # Create map visualization
        map_path = create_route_map(route_data, instructions)
        map_url = f"/map/{os.path.basename(map_path)}"

        # Store the map path temporarily
        app.state.temp_maps = getattr(app.state, "temp_maps", {})
        app.state.temp_maps[os.path.basename(map_path)] = map_path

        return RoutingResult(
            query=query_data.query,
            parsed_params=parsed_params,
            route_data=route_data,
            instructions=instructions,
            map_url=map_url,
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/map/{map_id}", response_class=HTMLResponse)
async def get_map(map_id: str):
    """Get a generated map.

    Args:
        map_id: The map ID

    Returns:
        HTMLResponse: The HTML map
    """
    temp_maps = getattr(app.state, "temp_maps", {})
    map_path = temp_maps.get(map_id)

    if not map_path or not os.path.exists(map_path):
        raise HTTPException(status_code=404, detail="Map not found")

    with open(map_path, "r") as f:
        map_html = f.read()

    return HTMLResponse(content=map_html)


def start_server():
    """Start the FastAPI server."""
    import uvicorn

    uvicorn.run(app, host=SERVICE_HOST, port=SERVICE_PORT, reload=RELOAD)
