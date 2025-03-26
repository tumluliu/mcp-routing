"""Geocoding service using Nominatim (OpenStreetMap)."""

import requests
from typing import Tuple, Optional, List, Dict, Any
import time
from loguru import logger
from .config import MUNICH_BBOX, MUNICH_EXTENDED_BBOX, MUNICH_CENTER
import re


class NominatimGeocoder:
    """Geocoding service using Nominatim (OpenStreetMap)."""

    def __init__(self, user_agent: str = "MCPRouting/1.0"):
        """Initialize the Nominatim geocoder.

        Args:
            user_agent: User agent string for Nominatim API
        """
        self.base_url = "https://nominatim.openstreetmap.org/search"
        self.reverse_url = "https://nominatim.openstreetmap.org/reverse"
        self.headers = {"User-Agent": user_agent}

        # Common tourism categories in OSM
        self.landmark_categories = [
            "tourism=attraction",
            "tourism=museum",
            "historic=monument",
            "leisure=park",
            "amenity=theatre",
            "amenity=marketplace",
        ]

        logger.info("Initialized NominatimGeocoder")

    def geocode(
        self, location: str, city: str = "Munich", country: str = "Germany"
    ) -> Optional[Tuple[float, float]]:
        """Convert a location string to coordinates using Nominatim.

        Args:
            location: Location string (address or landmark)
            city: City name to help with disambiguation (default: "Munich")
            country: Country name to help with disambiguation (default: "Germany")

        Returns:
            (latitude, longitude) tuple or None if not found
        """
        try:
            logger.debug(f"Geocoding location: {location}")

            # Clean the input location
            location = self._clean_location_string(location)
            location_lower = location.lower()

            # Check for special event/place patterns
            if self._is_special_location(location_lower):
                return self._handle_special_location(location_lower, city)

            # Generate query variations
            queries = self._generate_query_variations(location, city, country)

            # Add special case variations for potentially difficult queries
            if "oktoberfest" in location_lower or "wiesn" in location_lower:
                queries.insert(0, "Theresienwiese München")
                queries.insert(0, "Oktoberfest München")

            if (
                "english garden" in location_lower
                or "englischer garten" in location_lower
            ):
                queries.insert(0, "Englischer Garten München")

            if "olympic park" in location_lower or "olympiapark" in location_lower:
                queries.insert(0, "Olympiapark München")

            # Track attempted queries to avoid duplicates
            attempted_queries = set()

            # Try each query until we get a result
            for query in queries:
                if query in attempted_queries:
                    continue

                attempted_queries.add(query)
                logger.debug(f"Trying geocoding query: {query}")

                # First try with name search for better landmark results
                result = self._try_name_search(query)
                if result:
                    return result

                # Then try with general query parameters
                result = self._try_general_query(query)
                if result:
                    return result

                # Respect rate limiting
                time.sleep(1)

            # If no results from direct queries, try with landmark-specific search
            if self._might_be_landmark(location_lower):
                logger.debug(f"Trying landmark-specific search for: {location}")
                for category in self.landmark_categories:
                    result = self._try_landmark_search(location, city, category)
                    if result:
                        return result
                    time.sleep(0.5)  # Shorter delay for category searches

            # If we get here, all queries failed
            logger.warning(f"No results found for any query variation of: {location}")

            # Last resort: try to get approximate coordinates from the city itself
            if city.lower() == "munich" or city.lower() == "münchen":
                logger.warning(f"Using Munich city center as fallback for: {location}")
                return MUNICH_CENTER

            return None

        except Exception as e:
            logger.error(f"Error geocoding location '{location}': {str(e)}")
            return None

    def _try_name_search(self, query: str) -> Optional[Tuple[float, float]]:
        """Try a name-specific search which works better for landmarks.

        Args:
            query: The search query

        Returns:
            Coordinates tuple or None
        """
        try:
            params = {
                "q": query,
                "format": "json",
                "limit": 5,
                "namedetails": 1,  # Include name details
                "countrycodes": "de",
                "viewbox": f"{MUNICH_BBOX['min_lon']},{MUNICH_BBOX['min_lat']},{MUNICH_BBOX['max_lon']},{MUNICH_BBOX['max_lat']}",
                "bounded": 0,
            }

            response = requests.get(self.base_url, params=params, headers=self.headers)

            if response.status_code != 200:
                return None

            results = response.json()
            if not results:
                return None

            # Filter for results with name matching part of our query
            query_words = set(query.lower().split())
            matching_results = []

            for result in results:
                if "namedetails" in result:
                    name = result.get("namedetails", {}).get("name", "").lower()
                    name_words = set(name.split())
                    intersection = query_words.intersection(name_words)
                    if intersection:
                        matching_results.append(
                            (result, len(intersection) / len(query_words))
                        )

            # Sort by match quality
            if matching_results:
                matching_results.sort(key=lambda x: x[1], reverse=True)
                result = matching_results[0][0]
                lat = float(result["lat"])
                lon = float(result["lon"])
                if self._is_within_region(lat, lon):
                    logger.debug(
                        f"Found name match coordinates for '{query}': ({lat}, {lon})"
                    )
                    return (lat, lon)

            # If no matching results with namedetails, try the normal results
            for result in results:
                lat = float(result["lat"])
                lon = float(result["lon"])
                if self._is_within_region(lat, lon):
                    logger.debug(f"Found coordinates for '{query}': ({lat}, {lon})")
                    return (lat, lon)

            return None

        except Exception as e:
            logger.debug(f"Error in name search for '{query}': {str(e)}")
            return None

    def _try_general_query(self, query: str) -> Optional[Tuple[float, float]]:
        """Try a general query with standard parameters.

        Args:
            query: The search query

        Returns:
            Coordinates tuple or None
        """
        try:
            params = self._build_request_params(query)

            response = requests.get(self.base_url, params=params, headers=self.headers)

            if response.status_code != 200:
                logger.error(
                    f"Nominatim API error: {response.status_code} - {response.text}"
                )
                return None

            results = response.json()

            if not results:
                return None

            # Get the first result
            result = results[0]
            lat = float(result["lat"])
            lon = float(result["lon"])

            # Check if the result is within the extended Munich region
            if self._is_within_region(lat, lon):
                logger.debug(f"Found coordinates for '{query}': ({lat}, {lon})")
                return (lat, lon)
            else:
                logger.warning(
                    f"Result outside Munich region: ({lat}, {lon}) for query '{query}'"
                )
                # If we have multiple results, try the next ones
                for alt_result in results[1:3]:  # Check up to 2 more results
                    alt_lat = float(alt_result["lat"])
                    alt_lon = float(alt_result["lon"])
                    if self._is_within_region(alt_lat, alt_lon):
                        logger.debug(
                            f"Found alternative coordinates for '{query}': ({alt_lat}, {alt_lon})"
                        )
                        return (alt_lat, alt_lon)

            return None

        except Exception as e:
            logger.debug(f"Error in general query for '{query}': {str(e)}")
            return None

    def _try_landmark_search(
        self, location: str, city: str, category: str
    ) -> Optional[Tuple[float, float]]:
        """Try a landmark-specific search with OSM category.

        Args:
            location: The location string
            city: City name
            category: OSM category tag (e.g., "tourism=attraction")

        Returns:
            Coordinates tuple or None
        """
        try:
            category_key, category_value = category.split("=")

            params = {
                "q": f"{location}",
                "format": "json",
                "limit": 3,
                "addressdetails": 1,
                "countrycodes": "de",
                "viewbox": f"{MUNICH_BBOX['min_lon']},{MUNICH_BBOX['min_lat']},{MUNICH_BBOX['max_lon']},{MUNICH_BBOX['max_lat']}",
                "bounded": 0,
                category_key: category_value,  # Add category filter
            }

            response = requests.get(self.base_url, params=params, headers=self.headers)

            if response.status_code != 200:
                return None

            results = response.json()
            if not results:
                return None

            # Check each result
            for result in results:
                lat = float(result["lat"])
                lon = float(result["lon"])

                # Verify it's in Munich from address details
                address = result.get("address", {})
                result_city = address.get("city", "")
                if not result_city:
                    result_city = address.get("town", "")
                if not result_city:
                    result_city = address.get("county", "")

                if result_city.lower() in [
                    "munich",
                    "münchen",
                ] and self._is_within_region(lat, lon):
                    logger.debug(
                        f"Found landmark coordinates for '{location}' with category {category}: ({lat}, {lon})"
                    )
                    return (lat, lon)

            return None

        except Exception as e:
            logger.debug(
                f"Error in landmark search for '{location}' with category {category}: {str(e)}"
            )
            return None

    def _is_special_location(self, location: str) -> bool:
        """Check if the location is a special case requiring custom handling.

        Args:
            location: Lowercase location string

        Returns:
            True if special location, False otherwise
        """
        special_patterns = [
            r"oktoberfest",
            r"\bwiesn\b",
            r"^beer fest",
            r"^central station",
            r"^hauptbahnhof",
        ]

        return any(re.search(pattern, location) for pattern in special_patterns)

    def _handle_special_location(
        self, location: str, city: str
    ) -> Optional[Tuple[float, float]]:
        """Handle special location cases with dedicated search strategies.

        Args:
            location: Lowercase location string
            city: City name

        Returns:
            Coordinates tuple or None
        """
        # Oktoberfest - use structured search for Theresienwiese
        if "oktoberfest" in location or "wiesn" in location or "beer fest" in location:
            logger.info(f"Using special handling for Oktoberfest: {location}")
            return self._get_theresienwiese_coordinates()

        # Main station - use structured search
        if "central station" in location or "hauptbahnhof" in location:
            logger.info(f"Using special handling for main station: {location}")
            return self._get_main_station_coordinates()

        return None

    def _get_theresienwiese_coordinates(self) -> Optional[Tuple[float, float]]:
        """Get coordinates for Theresienwiese (Oktoberfest location) using structured search.

        Returns:
            Coordinates tuple or None
        """
        try:
            params = {
                "amenity": "fairground",
                "format": "json",
                "q": "Theresienwiese",
                "addressdetails": 1,
                "countrycodes": "de",
                "city": "München",
            }

            response = requests.get(self.base_url, params=params, headers=self.headers)

            if response.status_code != 200:
                logger.warning("Failed to get Theresienwiese with structured search")
                # Fallback to reverse geocoding known area
                return (
                    48.1351,
                    11.5494,
                )  # Still need these coordinates but as a fallback

            results = response.json()
            if results:
                lat = float(results[0]["lat"])
                lon = float(results[0]["lon"])
                logger.info(f"Found Theresienwiese at: ({lat}, {lon})")
                return (lat, lon)

            logger.warning("No results for Theresienwiese, using fallback coordinates")
            return (48.1351, 11.5494)  # Fallback coordinates

        except Exception as e:
            logger.error(f"Error getting Theresienwiese coordinates: {str(e)}")
            return (48.1351, 11.5494)  # Fallback coordinates

    def _get_main_station_coordinates(self) -> Optional[Tuple[float, float]]:
        """Get coordinates for Munich Main Station using structured search.

        Returns:
            Coordinates tuple or None
        """
        try:
            params = {
                "railway": "station",
                "format": "json",
                "q": "Hauptbahnhof München",
                "addressdetails": 1,
                "countrycodes": "de",
            }

            response = requests.get(self.base_url, params=params, headers=self.headers)

            if response.status_code != 200:
                logger.warning("Failed to get Hauptbahnhof with structured search")
                # Fallback to reverse geocoding known area
                return (48.1402, 11.5601)  # Fallback coordinates

            results = response.json()
            if results:
                lat = float(results[0]["lat"])
                lon = float(results[0]["lon"])
                logger.info(f"Found Hauptbahnhof at: ({lat}, {lon})")
                return (lat, lon)

            logger.warning("No results for Hauptbahnhof, using fallback coordinates")
            return (48.1402, 11.5601)  # Fallback coordinates

        except Exception as e:
            logger.error(f"Error getting Hauptbahnhof coordinates: {str(e)}")
            return (48.1402, 11.5601)  # Fallback coordinates

    def _clean_location_string(self, location: str) -> str:
        """Clean and normalize a location string.

        Args:
            location: Raw location string

        Returns:
            Cleaned location string
        """
        # Replace multiple spaces
        location = re.sub(r"\s+", " ", location).strip()

        # Remove common prefixes that aren't part of the location name
        prefixes_to_remove = ["go to ", "get to ", "arrive at ", "reach ", "visit "]
        for prefix in prefixes_to_remove:
            if location.lower().startswith(prefix):
                location = location[len(prefix) :]

        # Remove trailing punctuation
        location = location.rstrip(",.;:")

        return location

    def _generate_query_variations(
        self, location: str, city: str, country: str
    ) -> List[str]:
        """Generate different query variations for better geocoding results.

        Args:
            location: Location string
            city: City name
            country: Country name

        Returns:
            List of query variations
        """
        queries = []

        # 1. Original location without any modification
        queries.append(location)

        # Add variations with city/country
        location_lower = location.lower()

        # 2. Add city if not already present
        if city.lower() not in location_lower:
            queries.append(f"{location}, {city}")

        # 3. With country
        if country.lower() not in location_lower:
            queries.append(f"{location}, {country}")

        # 4. With city and country
        if city.lower() not in location_lower and country.lower() not in location_lower:
            queries.append(f"{location}, {city}, {country}")

        # 5. Handle special cases like "city center", "downtown", etc.
        if any(
            term in location_lower
            for term in ["downtown", "city center", "center", "centre", "central"]
        ):
            queries.append(f"{city} center")
            queries.append(f"center of {city}")
            queries.append(f"central {city}")

        # 6. Try with the city as a prefix
        if not location_lower.startswith(city.lower()):
            queries.append(f"{city} {location}")

        # 7. Add POI-specific variations for tourist locations
        location_words = location_lower.split()
        if any(
            word
            in [
                "park",
                "garden",
                "museum",
                "palace",
                "castle",
                "church",
                "cathedral",
                "theater",
                "theatre",
            ]
            for word in location_words
        ):
            # Try with tourism prefix for better results
            queries.append(f"tourism {location}, {city}")

        # 8. Add stadium/arena variations
        if (
            "arena" in location_lower
            or "stadium" in location_lower
            or "olympia" in location_lower
        ):
            queries.append(f"stadium {location}, {city}")
            queries.append(f"sports venue {location}, {city}")

        # Extract street address if present
        street_address = self._extract_street_address(location)
        if street_address:
            queries.append(street_address)
            queries.append(f"{street_address}, {city}")
            queries.append(f"{street_address}, {city}, {country}")

        # Remove duplicates while preserving order
        unique_queries = []
        for query in queries:
            if query not in unique_queries:
                unique_queries.append(query)

        return unique_queries

    def _build_request_params(self, query: str) -> Dict[str, Any]:
        """Build request parameters for Nominatim API.

        Args:
            query: Search query

        Returns:
            Dictionary of request parameters
        """
        params = {
            "q": query,
            "format": "json",
            "limit": 5,
            "addressdetails": 1,
            "countrycodes": "de",
            "accept-language": "en",  # Use English for consistent results
        }

        # Add viewbox parameter to prioritize results in Munich area
        params["viewbox"] = (
            f"{MUNICH_BBOX['min_lon']},{MUNICH_BBOX['min_lat']},{MUNICH_BBOX['max_lon']},{MUNICH_BBOX['max_lat']}"
        )

        # Don't restrict to the viewbox completely
        params["bounded"] = 0

        return params

    def _is_within_region(self, lat: float, lon: float, extended: bool = True) -> bool:
        """Check if coordinates are within Munich region.

        Args:
            lat: Latitude
            lon: Longitude
            extended: Whether to use extended region bounds

        Returns:
            True if within region, False otherwise
        """
        bbox = MUNICH_EXTENDED_BBOX if extended else MUNICH_BBOX

        return (
            bbox["min_lat"] <= lat <= bbox["max_lat"]
            and bbox["min_lon"] <= lon <= bbox["max_lon"]
        )

    def _extract_street_address(self, location: str) -> Optional[str]:
        """Extract street address from a location string.

        Args:
            location: Location string

        Returns:
            Street address or None if not found
        """
        # Simple pattern for street addresses in German format
        # Like "Leopoldstraße 12" or "Marienplatz 8"
        street_address_pattern = r"([A-Za-zäöüÄÖÜß\s\-\.]+)\s+(\d+[\w-]*)"
        match = re.search(street_address_pattern, location)

        if match:
            return match.group(0)
        return None

    def _might_be_landmark(self, location: str) -> bool:
        """Check if the location might be a landmark based on common terms.

        Args:
            location: Lowercase location string

        Returns:
            True if likely a landmark, False otherwise
        """
        landmark_terms = [
            "park",
            "garden",
            "museum",
            "palace",
            "castle",
            "church",
            "theatre",
            "theater",
            "stadium",
            "arena",
            "gallery",
            "monument",
            "memorial",
            "square",
            "platz",
            "markt",
            "tower",
            "bridge",
        ]

        return any(term in location for term in landmark_terms)
