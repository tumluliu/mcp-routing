"""Command-line interface for the MCP Routing service."""

import argparse
import json
import os
import sys
from typing import Dict, Any, Optional

from .api import start_server
from .llm import DeepSeekLLM
from .routing import get_routing_engine
from .visualization import create_route_map, display_map
from .config import ROUTING_ENGINE, DEEPSEEK_MODEL


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="MCP Routing Service")

    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Server command
    server_parser = subparsers.add_parser("server", help="Start the API server")
    server_parser.add_argument("--host", help="Host to bind to")
    server_parser.add_argument("--port", type=int, help="Port to bind to")
    server_parser.add_argument(
        "--reload", action="store_true", help="Enable reload mode"
    )

    # Route command
    route_parser = subparsers.add_parser("route", help="Process a routing query")
    route_parser.add_argument("query", help="Natural language routing query")
    route_parser.add_argument(
        "--engine", choices=["osrm", "openrouteservice"], help="Routing engine to use"
    )
    route_parser.add_argument("--model", help="DeepSeek model to use")
    route_parser.add_argument("--output", help="Output file for route data")
    route_parser.add_argument(
        "--display-map", action="store_true", help="Display the route map"
    )

    return parser.parse_args()


def process_route_query(
    query: str,
    engine_name: Optional[str] = None,
    model_name: Optional[str] = None,
    output_file: Optional[str] = None,
    display_map_flag: bool = False,
) -> Dict[str, Any]:
    """Process a routing query from the command line.

    Args:
        query: Natural language routing query
        engine_name: Routing engine name
        model_name: DeepSeek model name
        output_file: Output file for route data
        display_map_flag: Whether to display the map

    Returns:
        Dict with routing results
    """
    # Initialize services
    llm = DeepSeekLLM(model=model_name or DEEPSEEK_MODEL)
    routing_engine = get_routing_engine(engine_name or ROUTING_ENGINE)

    # Parse the query
    print("Parsing query with DeepSeek...")
    parsed_params = llm.parse_routing_query(query)

    print(f"Origin: {parsed_params.get('origin', 'Unknown')}")
    print(f"Destination: {parsed_params.get('destination', 'Unknown')}")
    print(f"Mode: {parsed_params.get('mode', 'driving')}")

    if parsed_params.get("waypoints"):
        print(f"Waypoints: {', '.join(str(wp) for wp in parsed_params['waypoints'])}")

    # Get the route
    print("\nFetching route...")
    route_data = routing_engine.route(
        origin=parsed_params["origin"],
        destination=parsed_params["destination"],
        mode=parsed_params.get("mode", "driving"),
        waypoints=parsed_params.get("waypoints"),
        avoid=parsed_params.get("avoid"),
    )

    # Generate navigation instructions
    print("Generating navigation instructions...")
    instructions = llm.generate_navigation_instructions(route_data)

    # Print route summary
    distance_km = route_data["distance"] / 1000
    duration_min = route_data["duration"] / 60

    print("\nRoute Summary:")
    print(f"Distance: {distance_km:.1f} km")
    print(f"Duration: {duration_min:.0f} min")

    # Print instructions
    print("\nNavigation Instructions:")
    for i, instruction in enumerate(instructions, 1):
        print(f"{i}. {instruction}")

    # Create map
    print("\nCreating route map...")
    map_path = create_route_map(route_data, instructions)
    print(f"Map saved to: {map_path}")

    # Display the map if requested
    if display_map_flag:
        print("Opening map in browser...")
        display_map(map_path)

    # Save output if requested
    if output_file:
        output_data = {
            "query": query,
            "parsed_params": parsed_params,
            "route_data": {
                "distance": route_data["distance"],
                "duration": route_data["duration"],
                "steps": route_data["steps"],
                "origin": route_data["origin"],
                "destination": route_data["destination"],
                # Exclude geometry to keep file size reasonable
            },
            "instructions": instructions,
            "map_path": map_path,
        }

        with open(output_file, "w") as f:
            json.dump(output_data, f, indent=2)

        print(f"Route data saved to: {output_file}")

    return {
        "query": query,
        "parsed_params": parsed_params,
        "route_data": route_data,
        "instructions": instructions,
        "map_path": map_path,
    }


def main():
    """Main entry point for the CLI."""
    args = parse_args()

    if args.command == "server":
        # Update environment variables if provided
        if args.host:
            os.environ["SERVICE_HOST"] = args.host
        if args.port:
            os.environ["SERVICE_PORT"] = str(args.port)
        if args.reload:
            os.environ["RELOAD"] = "True"

        print("Starting MCP Routing Service API server...")
        start_server()

    elif args.command == "route":
        process_route_query(
            query=args.query,
            engine_name=args.engine,
            model_name=args.model,
            output_file=args.output,
            display_map_flag=args.display_map,
        )

    else:
        print("Please specify a command. Use --help for more information.")
        sys.exit(1)


if __name__ == "__main__":
    main()
