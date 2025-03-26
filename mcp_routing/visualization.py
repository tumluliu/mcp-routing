"""Map visualization for routing results."""

import folium
from typing import Dict, List, Any
import tempfile
import webbrowser
import os


def create_route_map(
    route_data: Dict[str, Any], instructions: List[str] = None, save_path: str = None
) -> str:
    """Create a map visualization of a route.

    Args:
        route_data: Standardized routing data
        instructions: Navigation instructions
        save_path: Path to save the HTML map file

    Returns:
        Path to the saved HTML map file
    """
    # Extract route information
    origin = route_data["origin"]
    destination = route_data["destination"]
    geometry = route_data["geometry"]

    # Center the map at the midpoint of the route
    lat_sum = sum(point[0] for point in geometry)
    lon_sum = sum(point[1] for point in geometry)
    center_lat = lat_sum / len(geometry)
    center_lon = lon_sum / len(geometry)

    # Create a map
    m = folium.Map(location=[center_lat, center_lon], zoom_start=12)

    # Add the route as a polyline
    folium.PolyLine(
        locations=geometry, color="blue", weight=5, opacity=0.7, tooltip="Route"
    ).add_to(m)

    # Add markers for origin and destination
    folium.Marker(
        location=origin,
        popup="Origin",
        icon=folium.Icon(color="green", icon="play", prefix="fa"),
    ).add_to(m)

    folium.Marker(
        location=destination,
        popup="Destination",
        icon=folium.Icon(color="red", icon="stop", prefix="fa"),
    ).add_to(m)

    # Extract step information
    steps = route_data.get("steps", [])

    # Add navigation instructions if available
    if instructions:
        # Enhanced styling for navigation instructions
        instructions_html = """
        <div style="
            padding: 15px; 
            background-color: #f8f9fa; 
            border-radius: 8px; 
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
            font-family: Arial, sans-serif;
            max-height: 300px;
            overflow-y: auto;
            margin: 10px 0;
        ">
            <h4 style="
                color: #3366cc; 
                border-bottom: 1px solid #e1e4e8; 
                padding-bottom: 8px; 
                margin-top: 0;
            ">Navigation Instructions</h4>
            <ol style="
                padding-left: 25px;
                margin: 10px 0;
            ">
        """

        for i, instruction in enumerate(instructions):
            # Add different styling for different types of instructions (turn, continue, arrive)
            icon = "➡️"  # Default icon

            if any(kw in instruction.lower() for kw in ["turn left", "left onto"]):
                icon = "↩️"
            elif any(kw in instruction.lower() for kw in ["turn right", "right onto"]):
                icon = "↪️"
            elif any(kw in instruction.lower() for kw in ["straight", "continue"]):
                icon = "⬆️"
            elif any(kw in instruction.lower() for kw in ["destination", "arrive"]):
                icon = "🏁"
            elif any(kw in instruction.lower() for kw in ["uturn", "u-turn"]):
                icon = "↩️↩️"
            elif any(
                kw in instruction.lower() for kw in ["merge", "highway", "motorway"]
            ):
                icon = "🛣️"

            instructions_html += f"""
            <li style="
                margin-bottom: 8px;
                line-height: 1.5;
                color: #333;
            "><span style="margin-right: 5px;">{icon}</span> {instruction}</li>
            """

        instructions_html += """
            </ol>
        </div>
        """

        folium.Element(instructions_html).add_to(m)

    # Add step markers if there are more than 2 steps
    if len(steps) > 2:
        for i, step in enumerate(steps[1:-1], 1):  # Skip first and last
            # Only add markers for significant steps
            if step.get("distance", 0) > 50 and step.get("name", ""):
                # Sample point from the geometry
                sample_idx = int(i * len(geometry) / len(steps))
                if sample_idx < len(geometry):
                    point = geometry[sample_idx]

                    popup_text = f"""
                    <b>{step.get('instruction', '')}</b><br>
                    {step.get('name', '')}<br>
                    Distance: {step.get('distance', 0):.0f} m<br>
                    Time: {step.get('duration', 0):.0f} s
                    """

                    folium.Marker(
                        location=point,
                        popup=folium.Popup(popup_text, max_width=200),
                        icon=folium.Icon(color="blue", icon="info-sign", prefix="fa"),
                    ).add_to(m)

    # Add route summary
    distance_km = route_data.get("distance", 0) / 1000
    duration_min = route_data.get("duration", 0) / 60

    summary_html = f"""
    <div style="
        padding: 15px; 
        background-color: #f8f9fa;
        border-radius: 8px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        font-family: Arial, sans-serif;
        margin: 10px 0;
    ">
        <h4 style="
            color: #3366cc;
            border-bottom: 1px solid #e1e4e8;
            padding-bottom: 8px;
            margin-top: 0;
        ">Route Summary</h4>
        <div style="margin-top: 10px;">
            <b>Distance:</b> {distance_km:.1f} km<br>
            <b>Duration:</b> {duration_min:.0f} min<br>
        </div>
    </div>
    """

    folium.Element(summary_html).add_to(m)

    # Save the map
    if not save_path:
        temp_dir = tempfile.gettempdir()
        save_path = os.path.join(temp_dir, "route_map.html")

    m.save(save_path)
    return save_path


def display_map(map_path: str) -> None:
    """Open the map in a web browser.

    Args:
        map_path: Path to the HTML map file
    """
    webbrowser.open(f"file://{os.path.abspath(map_path)}", new=2)
