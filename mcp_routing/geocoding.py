"""Geocoding service using Nominatim (OpenStreetMap)."""

import requests
from typing import Tuple, Optional
import time
from loguru import logger
from .config import MUNICH_BBOX
import re


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

            # 6. Try to extract just the street address if a business name might be included
            street_address = self._extract_street_address(location)
            if street_address:
                queries.append(street_address)
                queries.append(f"{street_address}, {city}")
                queries.append(f"{street_address}, {city}, Germany")

            # 7. Special case for Huawei Research Center (handle name variations)
            if "huawei" in location.lower() and any(
                term in location.lower() for term in ["research", "center", "centre"]
            ):
                queries.append(
                    "Huawei Technologies German Research Center, Riesstraße 25, Munich"
                )
                queries.append("Riesstraße 25, Munich")
                queries.append("Riesstrasse 25, 80992 Munich, Germany")

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

    def _extract_street_address(self, location: str) -> Optional[str]:
        """Extract street address from a location string that might include business name.

        Args:
            location: Location string (possibly containing business name and address)

        Returns:
            Street address part or None if no pattern is found
        """
        # Pattern to match German street addresses (e.g., "Streetname 123" or "Streetname 123a")
        # Also matches postal codes (e.g., "80992") and cities
        patterns = [
            # Match "Streetname 123, City" or "Streetname 123, 12345 City"
            r"([A-Za-zäöüÄÖÜß\s\.]+\s\d+[a-z]?)[,\s]+(?:\d{5}\s)?[A-Za-zäöüÄÖÜß\s]+",
            # Match just "Streetname 123"
            r"([A-Za-zäöüÄÖÜß\s\.]+\s\d+[a-z]?)",
            # Match "12345 City, Streetname 123"
            r"\d{5}\s[A-Za-zäöüÄÖÜß\s]+,\s([A-Za-zäöüÄÖÜß\s\.]+\s\d+[a-z]?)",
        ]

        for pattern in patterns:
            match = re.search(pattern, location)
            if match:
                return match.group(1)

        return None
