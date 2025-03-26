"""Geocoding service using Nominatim (OpenStreetMap)."""

import requests
from typing import Tuple, Optional
import time
from loguru import logger
from .config import MUNICH_BBOX


class NominatimGeocoder:
    """Geocoding service using Nominatim (OpenStreetMap)."""

    def __init__(self, user_agent: str = "MCPRouting/1.0"):
        """Initialize the Nominatim geocoder.

        Args:
            user_agent: User agent string for Nominatim API
        """
        self.base_url = "https://nominatim.openstreetmap.org/search"
        self.headers = {"User-Agent": user_agent}
        logger.info("Initialized NominatimGeocoder")

    def geocode(
        self, location: str, city: str = "Munich"
    ) -> Optional[Tuple[float, float]]:
        """Convert a location string to coordinates using Nominatim.

        Args:
            location: Location string (address or landmark)
            city: City name to help with disambiguation (default: "Munich")

        Returns:
            (latitude, longitude) tuple or None if not found
        """
        try:
            logger.debug(f"Geocoding location: {location}")

            # Try different query structures
            queries = []

            # 1. Original location without any modification
            queries.append(location)

            # 2. Add city if not already present
            if city.lower() not in location.lower():
                queries.append(f"{location}, {city}")

            # 3. Add country to improve accuracy
            queries.append(f"{location}, Germany")

            # 4. With city and country
            if city.lower() not in location.lower():
                queries.append(f"{location}, {city}, Germany")

            # 5. Special case for airport queries - try explicit names
            if any(
                term in location.lower() for term in ["flughafen", "airport", "muc"]
            ):
                queries.append("Munich Airport")
                queries.append("Flughafen München")
                queries.append("Munich International Airport")

            # Try each query until we get a result
            for query in queries:
                logger.debug(f"Trying geocoding query: {query}")

                params = {
                    "q": query,
                    "format": "json",
                    "limit": 1,
                    "addressdetails": 1,
                    "countrycodes": "de",
                }

                # Only add viewbox and bounded parameters for location-specific queries
                if not any(term in query.lower() for term in ["airport", "flughafen"]):
                    params["viewbox"] = (
                        f"{MUNICH_BBOX['min_lon']},{MUNICH_BBOX['min_lat']},{MUNICH_BBOX['max_lon']},{MUNICH_BBOX['max_lat']}"
                    )
                    params["bounded"] = 1

                response = requests.get(
                    self.base_url, params=params, headers=self.headers
                )

                if response.status_code != 200:
                    logger.error(
                        f"Nominatim API error: {response.status_code} - {response.text}"
                    )
                    continue

                results = response.json()

                if not results:
                    continue

                # Get the first result
                result = results[0]
                lat = float(result["lat"])
                lon = float(result["lon"])

                # Check if the result is within a wider Munich region (extended bounding box)
                extended_bbox = {
                    "min_lat": 47.3,
                    "max_lat": 49.0,
                    "min_lon": 10.0,
                    "max_lon": 12.5,
                }

                if not (
                    extended_bbox["min_lat"] <= lat <= extended_bbox["max_lat"]
                    and extended_bbox["min_lon"] <= lon <= extended_bbox["max_lon"]
                ):
                    logger.warning(
                        f"Result outside extended Munich region: ({lat}, {lon})"
                    )
                    continue

                logger.debug(f"Found coordinates for '{query}': ({lat}, {lon})")
                return (lat, lon)

            # If we get here, all queries failed
            logger.warning(f"No results found for any query variation of: {location}")
            return None

        except Exception as e:
            logger.error(f"Error geocoding location '{location}': {str(e)}")
            return None

        finally:
            # Respect Nominatim's usage policy by adding a delay
            time.sleep(1)
