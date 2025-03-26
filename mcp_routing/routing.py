"""Routing engine interfaces for OSRM and OpenRouteService."""

import json
import requests
from typing import Dict, List, Optional, Any, Union, Tuple
import polyline

from .config import (
    ROUTING_ENGINE,
    OSRM_ENDPOINT,
    ORS_ENDPOINT,
    ORS_API_KEY,
    MUNICH_BBOX,
)


class RoutingEngine:
    """Base class for routing engines."""

    def geocode(self, location: str) -> Tuple[float, float]:
        """Convert a location string to coordinates.

        Args:
            location: Location string (address)

        Returns:
            (latitude, longitude) tuple
        """
        # In a real implementation, this would use a geocoding service
        # For this PoC, we'll use a simplified version that works with Munich's major locations

        # Dictionary of known Munich locations
        munich_locations = {
            "marienplatz": (48.1373, 11.5754),
            "hauptbahnhof": (48.1402, 11.5600),
            "olympiapark": (48.1698, 11.5516),
            "englischer garten": (48.1642, 11.6056),
            "allianz arena": (48.2188, 11.6248),
            "deutsches museum": (48.1299, 11.5834),
            "viktualienmarkt": (48.1348, 11.5765),
            "odeonsplatz": (48.1424, 11.5765),
            "frauenkirche": (48.1385, 11.5733),
            "bmw welt": (48.1771, 11.5562),
            "munich airport": (48.3537, 11.7750),
            "sendlinger tor": (48.1342, 11.5666),
            "karlsplatz": (48.1399, 11.5655),
            "technische universität münchen": (48.1497, 11.5680),
            "ludwig maximilians universität": (48.1507, 11.5801),
            "olympic stadium": (48.1731, 11.5465),
            "hofbräuhaus": (48.1379, 11.5797),
            "nymphenburg palace": (48.1583, 11.5033),
            "olympia einkaufszentrum": (48.1828, 11.5358),
            "tierpark hellabrunn": (48.0784, 11.5554),
        }

        # Simplistic lookup
        location_lower = location.lower()

        # Check for exact matches
        for name, coords in munich_locations.items():
            if name in location_lower or location_lower in name:
                return coords

        # If no match found, generate a pseudo-random point in Munich's bounding box
        # This is just for demonstration - a real implementation would use proper geocoding
        import hashlib
        import struct

        # Use a hash of the input to generate a deterministic but "random" point
        hash_val = hashlib.md5(location_lower.encode()).digest()
        lat_ratio, lon_ratio = struct.unpack("ff", hash_val[:8])

        # Ensure the values are between 0 and 1
        lat_ratio = abs(lat_ratio) % 1.0
        lon_ratio = abs(lon_ratio) % 1.0

        # Map to Munich's bounding box
        lat = MUNICH_BBOX["min_lat"] + lat_ratio * (
            MUNICH_BBOX["max_lat"] - MUNICH_BBOX["min_lat"]
        )
        lon = MUNICH_BBOX["min_lon"] + lon_ratio * (
            MUNICH_BBOX["max_lon"] - MUNICH_BBOX["min_lon"]
        )

        return (lat, lon)

    def route(
        self,
        origin: Union[str, Tuple[float, float]],
        destination: Union[str, Tuple[float, float]],
        **kwargs,
    ) -> Dict[str, Any]:
        """Get routing information between origin and destination.

        This is an abstract method that should be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses must implement route()")


class OSRMEngine(RoutingEngine):
    """OSRM routing engine interface."""

    def __init__(self, endpoint: str = OSRM_ENDPOINT):
        """Initialize OSRM interface.

        Args:
            endpoint: OSRM API endpoint
        """
        self.endpoint = endpoint

    def route(
        self,
        origin: Union[str, Tuple[float, float]],
        destination: Union[str, Tuple[float, float]],
        mode: str = "driving",
        waypoints: Optional[List[Union[str, Tuple[float, float]]]] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Get routing information from OSRM.

        Args:
            origin: Origin location (address or coordinates)
            destination: Destination location (address or coordinates)
            mode: Transportation mode (car, bike, foot)
            waypoints: List of waypoints
            **kwargs: Additional routing parameters

        Returns:
            Routing data
        """
        # Convert addresses to coordinates if needed
        if isinstance(origin, str):
            origin = self.geocode(origin)

        if isinstance(destination, str):
            destination = self.geocode(destination)

        # Process waypoints if any
        waypoint_coords = []
        if waypoints:
            for wp in waypoints:
                if isinstance(wp, str):
                    waypoint_coords.append(self.geocode(wp))
                else:
                    waypoint_coords.append(wp)

        # Build coordinates string (OSRM uses lon,lat order)
        coords = [f"{origin[1]},{origin[0]}"]

        if waypoint_coords:
            for wp in waypoint_coords:
                coords.append(f"{wp[1]},{wp[0]}")

        coords.append(f"{destination[1]},{destination[0]}")

        # Map mode to OSRM profile
        profile_map = {"driving": "car", "walking": "foot", "cycling": "bike"}
        profile = profile_map.get(mode.lower(), "car")

        # Build URL
        coords_str = ";".join(coords)
        url = f"{self.endpoint}/route/v1/{profile}/{coords_str}"

        # Add optional parameters
        params = {
            "overview": "full",
            "geometries": "polyline",
            "steps": "true",
            **{k: v for k, v in kwargs.items() if v is not None},
        }

        # Make request
        response = requests.get(url, params=params)

        if response.status_code != 200:
            raise Exception(f"OSRM API error: {response.status_code} - {response.text}")

        route_data = response.json()

        # Process and standardize the response
        return self._standardize_response(route_data, origin, destination)

    def _standardize_response(
        self,
        route_data: Dict[str, Any],
        origin: Tuple[float, float],
        destination: Tuple[float, float],
    ) -> Dict[str, Any]:
        """Convert OSRM response to standardized format.

        Args:
            route_data: OSRM response data
            origin: Origin coordinates
            destination: Destination coordinates

        Returns:
            Standardized routing data
        """
        if not route_data.get("routes"):
            raise Exception("No routes found in OSRM response")

        route = route_data["routes"][0]

        # Decode polyline
        geometry = polyline.decode(route["geometry"])
        # Convert to [lat, lon] format
        geometry = [(point[0], point[1]) for point in geometry]

        # Process steps
        steps = []
        for leg in route["legs"]:
            for step in leg["steps"]:
                steps.append(
                    {
                        "distance": step["distance"],
                        "duration": step["duration"],
                        "instruction": step["maneuver"]["type"],
                        "name": step.get("name", ""),
                    }
                )

        return {
            "distance": route["distance"],
            "duration": route["duration"],
            "geometry": geometry,
            "steps": steps,
            "origin": origin,
            "destination": destination,
        }


class OpenRouteServiceEngine(RoutingEngine):
    """OpenRouteService routing engine interface."""

    def __init__(self, endpoint: str = ORS_ENDPOINT, api_key: str = ORS_API_KEY):
        """Initialize OpenRouteService interface.

        Args:
            endpoint: OpenRouteService API endpoint
            api_key: OpenRouteService API key
        """
        self.endpoint = endpoint
        self.api_key = api_key
        self.headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        if api_key:
            self.headers["Authorization"] = api_key

    def route(
        self,
        origin: Union[str, Tuple[float, float]],
        destination: Union[str, Tuple[float, float]],
        mode: str = "driving",
        waypoints: Optional[List[Union[str, Tuple[float, float]]]] = None,
        avoid: Optional[List[str]] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Get routing information from OpenRouteService.

        Args:
            origin: Origin location (address or coordinates)
            destination: Destination location (address or coordinates)
            mode: Transportation mode (driving, cycling, walking)
            waypoints: List of waypoints
            avoid: Features to avoid (tolls, highways, ferries)
            **kwargs: Additional routing parameters

        Returns:
            Routing data
        """
        # Convert addresses to coordinates if needed
        if isinstance(origin, str):
            origin = self.geocode(origin)

        if isinstance(destination, str):
            destination = self.geocode(destination)

        # Process waypoints if any
        waypoint_coords = []
        if waypoints:
            for wp in waypoints:
                if isinstance(wp, str):
                    waypoint_coords.append(self.geocode(wp))
                else:
                    waypoint_coords.append(wp)

        # Build coordinates array (ORS uses [lon, lat] order)
        coordinates = [[origin[1], origin[0]]]

        if waypoint_coords:
            for wp in waypoint_coords:
                coordinates.append([wp[1], wp[0]])

        coordinates.append([destination[1], destination[0]])

        # Map mode to ORS profile
        profile_map = {
            "driving": "driving-car",
            "walking": "foot-walking",
            "cycling": "cycling-regular",
        }
        profile = profile_map.get(mode.lower(), "driving-car")

        # Map avoid parameters
        avoid_features = []
        if avoid:
            feature_map = {
                "tolls": "tollways",
                "highways": "highways",
                "ferries": "ferries",
            }
            avoid_features = [
                feature_map.get(item, item) for item in avoid if item in feature_map
            ]

        # Build request body
        payload = {
            "coordinates": coordinates,
            "instructions": True,
            "format": "geojson",
        }

        if avoid_features:
            payload["options"] = {"avoid_features": avoid_features}

        # Add other parameters if provided
        for key, value in kwargs.items():
            if value is not None:
                payload[key] = value

        # Make request
        url = f"{self.endpoint}/v2/directions/{profile}/geojson"
        response = requests.post(url, headers=self.headers, json=payload)

        if response.status_code != 200:
            raise Exception(
                f"OpenRouteService API error: {response.status_code} - {response.text}"
            )

        route_data = response.json()

        # Process and standardize the response
        return self._standardize_response(route_data, origin, destination)

    def _standardize_response(
        self,
        route_data: Dict[str, Any],
        origin: Tuple[float, float],
        destination: Tuple[float, float],
    ) -> Dict[str, Any]:
        """Convert OpenRouteService response to standardized format.

        Args:
            route_data: OpenRouteService response data
            origin: Origin coordinates
            destination: Destination coordinates

        Returns:
            Standardized routing data
        """
        if not route_data.get("features"):
            raise Exception("No routes found in OpenRouteService response")

        route = route_data["features"][0]
        properties = route["properties"]

        # Extract geometry (ORS uses GeoJSON format with [lon, lat] order)
        geometry_coords = route["geometry"]["coordinates"]
        # Convert to [lat, lon] format
        geometry = [(coord[1], coord[0]) for coord in geometry_coords]

        # Process steps
        steps = []
        if "segments" in properties:
            for segment in properties["segments"]:
                for step in segment.get("steps", []):
                    steps.append(
                        {
                            "distance": step.get("distance", 0),
                            "duration": step.get("duration", 0),
                            "instruction": step.get("instruction", ""),
                            "name": step.get("name", ""),
                        }
                    )

        return {
            "distance": properties.get("summary", {}).get("distance", 0),
            "duration": properties.get("summary", {}).get("duration", 0),
            "geometry": geometry,
            "steps": steps,
            "origin": origin,
            "destination": destination,
        }


def get_routing_engine(engine_name: str = ROUTING_ENGINE) -> RoutingEngine:
    """Factory function to get the appropriate routing engine.

    Args:
        engine_name: Name of the routing engine to use

    Returns:
        Routing engine instance
    """
    if engine_name.lower() == "osrm":
        return OSRMEngine()
    elif engine_name.lower() == "openrouteservice":
        return OpenRouteServiceEngine()
    elif engine_name.lower() == "dummy":
        return DummyRoutingEngine()
    else:
        # Default to DummyRoutingEngine if the specified engine is not supported
        print(
            f"Warning: Unsupported routing engine: {engine_name}. Using DummyRoutingEngine instead."
        )
        return DummyRoutingEngine()


class DummyRoutingEngine:
    """Dummy routing engine that returns fake data when actual routing engines fail."""

    def __init__(self):
        """Initialize the dummy routing engine."""
        print(
            "WARNING: Using DummyRoutingEngine. This should only be used for testing."
        )

    def route(
        self,
        origin,
        destination,
        mode="driving",
        waypoints=None,
        avoid=None,
        departure_time=None,
        arrival_time=None,
    ):
        """Return fake routing data.

        Args:
            origin: Origin location (address or coordinates)
            destination: Destination location (address or coordinates)
            mode: Transportation mode (default: "driving")
            waypoints: List of intermediate waypoints (optional)
            avoid: List of features to avoid (optional)
            departure_time: Departure time (optional)
            arrival_time: Arrival time (optional)

        Returns:
            Dummy routing data in standardized format
        """
        # Munich center coordinates (Marienplatz)
        munich_center = [48.1371, 11.5754]

        # Fake route geometry (circular route around Munich center)
        geometry = []
        for i in range(21):
            angle = i * 0.1
            lat = munich_center[0] + 0.01 * angle * (1 if i % 2 == 0 else -1)
            lon = munich_center[1] + 0.01 * angle * (1 if i % 3 == 0 else -1)
            geometry.append([lat, lon])

        # Fake steps for the route
        steps = [
            {
                "name": "Start Street",
                "instruction": "Start from origin",
                "distance": 100,
                "duration": 60,
            },
            {
                "name": "Main Street",
                "instruction": "Continue on Main Street",
                "distance": 500,
                "duration": 300,
            },
            {
                "name": "Central Avenue",
                "instruction": "Turn right onto Central Avenue",
                "distance": 800,
                "duration": 480,
            },
            {
                "name": "Destination Road",
                "instruction": "Arrive at destination",
                "distance": 200,
                "duration": 120,
            },
        ]

        # Process origin and destination to handle different input formats
        if isinstance(origin, (list, tuple)):
            origin_coords = origin[:2]
        else:
            # For string addresses, use dummy coordinates near Munich center
            origin_coords = [munich_center[0] - 0.01, munich_center[1] - 0.01]

        if isinstance(destination, (list, tuple)):
            dest_coords = destination[:2]
        else:
            # For string addresses, use dummy coordinates near Munich center
            dest_coords = [munich_center[0] + 0.01, munich_center[1] + 0.01]

        # Return standardized route data
        return {
            "origin": origin_coords,
            "destination": dest_coords,
            "distance": 1600,  # meters
            "duration": 960,  # seconds
            "mode": mode,
            "geometry": geometry,
            "steps": steps,
            "waypoints": waypoints or [],
            "summary": "Dummy route from origin to destination",
            "bounds": {
                "min_lat": min(p[0] for p in geometry),
                "max_lat": max(p[0] for p in geometry),
                "min_lon": min(p[1] for p in geometry),
                "max_lon": max(p[1] for p in geometry),
            },
        }
