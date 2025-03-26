"""Place search service for finding businesses and POIs."""

import os
import json
import requests
from typing import List, Dict, Any, Optional, Tuple
from loguru import logger
import time
import random
import re
from pathlib import Path
import math

from .geocoding import NominatimGeocoder
from .config import MUNICH_BBOX, MUNICH_CENTER, DEFAULT_SEARCH_RADIUS

# Optional: Define environment variable for Overpass API endpoint
OVERPASS_API_ENDPOINT = os.getenv(
    "OVERPASS_API_ENDPOINT", "https://overpass-api.de/api/interpreter"
)

# Create a cache directory for place data
CACHE_DIR = Path("cache/places")
CACHE_DIR.mkdir(parents=True, exist_ok=True)


class PlaceSearchService:
    """Service for searching places by category near a location."""

    def __init__(self, use_cache: bool = True):
        """Initialize the place search service.

        Args:
            use_cache: Whether to use cache for place data
        """
        self.geocoder = NominatimGeocoder()
        self.use_cache = use_cache
        logger.info("Initialized PlaceSearchService")

        # Define mappings from natural language categories to OSM tags
        self.category_mappings = {
            "restaurant": {"amenity": ["restaurant"]},
            "cafe": {"amenity": ["cafe"]},
            "bar": {"amenity": ["bar", "pub", "biergarten"]},
            "coffee": {"amenity": ["cafe"]},
            "fast food": {"amenity": ["fast_food"]},
            "hotel": {"tourism": ["hotel"]},
            "supermarket": {"shop": ["supermarket"]},
            "bakery": {"shop": ["bakery"]},
            "museum": {"tourism": ["museum"]},
            "cinema": {"amenity": ["cinema"]},
            "theater": {"amenity": ["theatre"]},
            "park": {"leisure": ["park"]},
            "atm": {"amenity": ["atm"]},
            "bank": {"amenity": ["bank"]},
            "pharmacy": {"amenity": ["pharmacy"]},
            "hospital": {"amenity": ["hospital"]},
            "school": {"amenity": ["school"]},
            "university": {"amenity": ["university"]},
            "gym": {"leisure": ["fitness_centre"]},
            "parking": {"amenity": ["parking"]},
            "gas station": {"amenity": ["fuel"]},
            "gas": {"amenity": ["fuel"]},
            "petrol": {"amenity": ["fuel"]},
            "swimming pool": {"leisure": ["swimming_pool"]},
            "tourist attraction": {"tourism": ["attraction"]},
        }

        # Define important landmarks with their coordinates for contextual ranking
        self.landmarks = {
            "isar": (48.1255, 11.5827),  # Central point of the Isar river
            "oktoberfest": (48.1351, 11.5494),  # Theresienwiese
            "marienplatz": (48.1371, 11.5754),  # City center
            "english_garden": (48.1642, 11.6047),  # English Garden
            "olympiapark": (48.1756, 11.5521),  # Olympic Park
            "hauptbahnhof": (48.1402, 11.5601),  # Main Station
        }

        # Cultural district/neighborhood weights for local context
        self.districts = {
            "schwabing": {"nightlife": 0.8, "arts": 0.9, "dining": 0.7},
            "glockenbachviertel": {"nightlife": 0.9, "lgbtq": 0.9, "dining": 0.8},
            "haidhausen": {"nightlife": 0.7, "arts": 0.6, "dining": 0.7},
            "maxvorstadt": {"arts": 0.8, "students": 0.9, "dining": 0.7},
            "au": {"nightlife": 0.6, "authentic": 0.8, "dining": 0.7},
            "sendling": {"authentic": 0.7, "local": 0.8},
            "neuhausen": {"family": 0.7, "local": 0.7},
        }

        # Popularity adjustment based on Munich local knowledge
        self.local_knowledge = {
            # Popular places with locals get a boost
            "augustiner": 1.3,
            "paulaner": 1.2,
            "hofbräu": 1.2,
            "löwenbräu": 1.2,
            "schneider": 1.2,
            "wiener": 1.2,
            "biergarten": 1.3,
            "brauhaus": 1.2,
            "zum": 1.1,  # Common prefix for traditional places
            "alte": 1.1,  # "Old" often indicates traditional places
            "gast": 1.1,  # Part of "Gasthaus", traditional inn
            "keller": 1.2,  # Traditional beer cellars
        }

    def search_nearby(
        self,
        category: str,
        near_location: str,
        radius: int = 1000,
        limit: int = 3,
        extras: Optional[Dict[str, Any]] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """Search for places of a certain category near a location.

        Args:
            category: Type of place to search for (restaurant, cafe, etc.)
            near_location: Location to search near
            radius: Search radius in meters
            limit: Maximum number of results
            extras: Optional extra parameters for filtering (e.g. {"cuisine": "italian"})
            context: Optional contextual information (landmarks, time of day, etc.)

        Returns:
            List of places with coordinates and details
        """
        logger.info(f"Searching for {category} near {near_location} within {radius}m")

        # Process context if provided or initialize default context
        context = self._process_context(context, category, near_location)

        # Check if we have cached results
        if self.use_cache and not context.get("skip_cache", False):
            cache_key = f"{category}_{near_location}_{radius}_{limit}"
            cache_file = CACHE_DIR / f"{cache_key.replace(' ', '_')}.json"

            if cache_file.exists():
                try:
                    with open(cache_file, "r") as f:
                        cached_results = json.load(f)
                        logger.info(f"Using cached results for {cache_key}")
                        # Even with cached results, apply contextual ranking
                        refined_results = self._apply_contextual_ranking(
                            cached_results, context
                        )
                        return refined_results[:limit]
                except Exception as e:
                    logger.warning(f"Failed to load cache for {cache_key}: {e}")

        # 1. Geocode the reference location
        coords = self.geocoder.geocode(near_location)
        if not coords:
            logger.error(f"Failed to geocode reference location: {near_location}")
            return []

        lat, lon = coords
        logger.debug(f"Geocoded {near_location} to ({lat}, {lon})")

        # Store reference coordinates in context
        context["reference_coords"] = (lat, lon)

        # 2. Map natural language category to OSM tags
        osm_tags = self._get_osm_tags_for_category(category)
        if not osm_tags:
            logger.warning(f"No OSM tag mapping for category: {category}")
            # Fallback to amenity=restaurant for food-related queries
            if any(
                word in category.lower()
                for word in ["food", "eat", "restaurant", "dining"]
            ):
                osm_tags = {"amenity": ["restaurant"]}
            elif any(
                word in category.lower()
                for word in ["drink", "pub", "bar", "beer", "alcohol"]
            ):
                osm_tags = {"amenity": ["bar", "pub", "biergarten"]}
            else:
                # Try to infer from keywords
                osm_tags = self._infer_osm_tags(category)

        # 3. Use Overpass API to search for places
        try:
            # For context-aware searches, use larger radius to get more candidates
            search_radius = radius
            if context.get("nearby_landmark"):
                # Use larger radius when looking near landmarks to get more options
                search_radius = radius * 1.5

            # Use dynamic limit based on context
            search_limit = limit * 3  # Get more results for better ranking
            places = self._search_with_overpass(
                lat, lon, search_radius, osm_tags, search_limit, extras
            )

            # If we're looking near Oktoberfest or Isar and didn't find enough places,
            # try with specific search strategies
            if len(places) < limit and any(
                landmark in near_location.lower()
                for landmark in [
                    "oktoberfest",
                    "wiesn",
                    "theresienwiese",
                    "isar",
                    "river",
                ]
            ):
                logger.info(f"Using enhanced search for {near_location}")
                places = self._enhanced_landmark_search(
                    category,
                    near_location,
                    search_radius,
                    search_limit,
                    extras,
                    context,
                )

        except Exception as e:
            logger.error(f"Error searching with Overpass API: {e}")
            places = self._generate_backup_places(
                category, lat, lon, radius, limit, context
            )

        # Calculate distance and apply initial sort by distance
        for place in places:
            place["distance"] = self._calculate_distance(
                lat, lon, place["coordinates"][0], place["coordinates"][1]
            )

        # Apply sophisticated ranking considering context
        ranked_places = self._apply_contextual_ranking(places, context)

        # Limit results
        result = ranked_places[:limit]

        # Add contextual relevance information
        for place in result:
            if "relevance_factors" not in place and "relevance_score" in place:
                place["relevance_factors"] = self._explain_relevance(place, context)

        # Cache the results if caching is enabled
        if self.use_cache and result:
            try:
                with open(cache_file, "w") as f:
                    json.dump(result, f)
                    logger.debug(f"Cached results for {cache_key}")
            except Exception as e:
                logger.warning(f"Failed to cache results for {cache_key}: {e}")

        return result

    def _process_context(
        self, context: Optional[Dict[str, Any]], category: str, near_location: str
    ) -> Dict[str, Any]:
        """Process and enrich the context information.

        Args:
            context: Original context or None
            category: Search category
            near_location: Location string

        Returns:
            Enriched context dictionary
        """
        if context is None:
            context = {}

        # Initialize if not present
        if "nearby_landmarks" not in context:
            context["nearby_landmarks"] = []

        # Detect landmarks in the location string
        near_location_lower = near_location.lower()
        for landmark_name, coords in self.landmarks.items():
            # Check common variations of landmark names
            variations = [landmark_name]
            if landmark_name == "isar":
                variations.extend(["isar river", "isarfluss", "fluss", "river"])
            elif landmark_name == "oktoberfest":
                variations.extend(["wiesn", "theresienwiese", "beer festival"])
            elif landmark_name == "english_garden":
                variations.extend(["englischer garten", "park", "garden"])

            # Check if any variation is in the location string
            if any(variation in near_location_lower for variation in variations):
                context["nearby_landmarks"].append(
                    {"name": landmark_name, "coordinates": coords, "mentioned": True}
                )
                context["nearby_landmark"] = landmark_name

        # Extract time of day if available
        if "time" not in context:
            # Default to midday if not specified
            context["time"] = "midday"

        # Detect category-specific contexts
        if category.lower() in ["bar", "pub", "nightclub", "club"]:
            context["activity"] = "nightlife"
        elif category.lower() in ["restaurant", "cafe", "bistro"]:
            context["activity"] = "dining"

        return context

    def _enhanced_landmark_search(
        self,
        category: str,
        near_location: str,
        radius: int,
        limit: int,
        extras: Optional[Dict[str, Any]],
        context: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """Perform an enhanced search for places near important landmarks.

        Args:
            category: Type of place to search for
            near_location: Location string
            radius: Search radius
            limit: Max results
            extras: Extra search parameters
            context: Contextual information

        Returns:
            List of places
        """
        places = []
        reference_coords = context.get("reference_coords")

        # Identify which landmark to use for searching
        landmark_coords = None
        if "isar" in near_location.lower():
            landmark_coords = self.landmarks["isar"]
            logger.info("Using Isar river as landmark for enhanced search")
        elif any(
            term in near_location.lower()
            for term in ["oktoberfest", "wiesn", "theresienwiese"]
        ):
            landmark_coords = self.landmarks["oktoberfest"]
            logger.info("Using Oktoberfest location for enhanced search")

        if landmark_coords and reference_coords:
            # Try searching around the landmark
            osm_tags = self._get_osm_tags_for_category(category)

            # First, try directly at the landmark with a smaller radius
            landmark_places = self._search_with_overpass(
                landmark_coords[0],
                landmark_coords[1],
                radius * 0.7,
                osm_tags,
                limit,
                extras,
            )

            # Calculate distances both to reference and landmark
            for place in landmark_places:
                place["distance"] = self._calculate_distance(
                    reference_coords[0],
                    reference_coords[1],
                    place["coordinates"][0],
                    place["coordinates"][1],
                )
                place["landmark_distance"] = self._calculate_distance(
                    landmark_coords[0],
                    landmark_coords[1],
                    place["coordinates"][0],
                    place["coordinates"][1],
                )
                # Add a factor for proximity to the landmark
                place["landmark_factor"] = 1.0 - min(
                    1.0, place["landmark_distance"] / (radius * 0.7)
                )

            # For Isar, try a linear search along the river
            if "isar" in near_location.lower():
                # Multiple points along the Isar to search
                isar_points = [
                    (48.1040, 11.5560),  # Southern Isar
                    (48.1151, 11.5650),  # Near Flaucher area
                    (48.1255, 11.5827),  # Central Isar
                    (48.1350, 11.5860),  # Northern central Isar
                    (48.1450, 11.5927),  # Northern Isar
                ]

                for isar_point in isar_points:
                    point_places = self._search_with_overpass(
                        isar_point[0],
                        isar_point[1],
                        radius * 0.5,
                        osm_tags,
                        limit // 2,
                        extras,
                    )

                    # Calculate distances
                    for place in point_places:
                        place["distance"] = self._calculate_distance(
                            reference_coords[0],
                            reference_coords[1],
                            place["coordinates"][0],
                            place["coordinates"][1],
                        )
                        place["landmark_distance"] = self._calculate_distance(
                            isar_point[0],
                            isar_point[1],
                            place["coordinates"][0],
                            place["coordinates"][1],
                        )
                        place["landmark_factor"] = 1.0 - min(
                            1.0, place["landmark_distance"] / (radius * 0.5)
                        )

                    # Merge results avoiding duplicates
                    for place in point_places:
                        if not any(
                            existing["name"] == place["name"] for existing in places
                        ):
                            places.append(place)

            # Merge with landmark places avoiding duplicates
            for place in landmark_places:
                if not any(existing["name"] == place["name"] for existing in places):
                    places.append(place)

        # If we still don't have enough places, try a broader search
        if len(places) < limit:
            # Fall back to standard search with larger radius
            osm_tags = self._get_osm_tags_for_category(category)
            broader_places = self._search_with_overpass(
                reference_coords[0],
                reference_coords[1],
                radius * 2,
                osm_tags,
                limit * 2,
                extras,
            )

            # Calculate distances
            for place in broader_places:
                place["distance"] = self._calculate_distance(
                    reference_coords[0],
                    reference_coords[1],
                    place["coordinates"][0],
                    place["coordinates"][1],
                )

            # Merge results avoiding duplicates
            for place in broader_places:
                if not any(existing["name"] == place["name"] for existing in places):
                    places.append(place)

        return places

    def _apply_contextual_ranking(
        self, places: List[Dict[str, Any]], context: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Apply sophisticated ranking based on contextual factors.

        Args:
            places: List of places to rank
            context: Contextual information for ranking

        Returns:
            Ranked list of places
        """
        if not places:
            return []

        # Define weight factors for different criteria
        weights = {
            "distance": 0.35,  # Base weight for distance
            "landmark_proximity": 0.25,  # Weight for proximity to mentioned landmarks
            "local_popularity": 0.20,  # Weight for local popularity
            "cuisine_match": 0.10,  # Weight for cuisine match if applicable
            "ratings": 0.10,  # Weight for ratings if available
        }

        # Adjust weights based on context
        if context.get("nearby_landmark"):
            # If a specific landmark was mentioned, increase its importance
            weights["landmark_proximity"] = 0.40
            weights["distance"] = 0.25

        # For nightlife activities, local popularity is more important
        if context.get("activity") == "nightlife":
            weights["local_popularity"] = 0.30
            weights["distance"] = 0.25

        # Calculate maximum distance for normalization
        max_distance = max(place.get("distance", 0) for place in places) or 1

        # Process each place
        for place in places:
            # Initialize scores dictionary
            scores = {}

            # 1. Distance score (inversely proportional to distance)
            distance = place.get("distance", max_distance)
            scores["distance"] = 1.0 - (distance / max_distance)

            # 2. Landmark proximity (if applicable)
            scores["landmark_proximity"] = 0.0
            if context.get("nearby_landmarks"):
                for landmark in context["nearby_landmarks"]:
                    landmark_coords = landmark["coordinates"]
                    place_coords = place["coordinates"]

                    # Calculate distance to landmark
                    landmark_distance = self._calculate_distance(
                        landmark_coords[0],
                        landmark_coords[1],
                        place_coords[0],
                        place_coords[1],
                    )

                    # Normalize based on search radius (consider closer is better)
                    landmark_score = 1.0 - min(1.0, landmark_distance / 1000.0)

                    # If this landmark was specifically mentioned, give it more weight
                    if landmark.get("mentioned", False):
                        landmark_score *= 1.5

                    # Update the score with the highest landmark score
                    scores["landmark_proximity"] = max(
                        scores["landmark_proximity"], landmark_score
                    )

            # 3. Local popularity based on name matching with local knowledge
            scores["local_popularity"] = 0.5  # Base score
            place_name_lower = place.get("name", "").lower()

            for key, boost in self.local_knowledge.items():
                if key in place_name_lower:
                    scores["local_popularity"] = min(
                        1.0, scores["local_popularity"] * boost
                    )
                    place["local_match"] = key
                    break

            # 4. Cuisine match if applicable
            scores["cuisine_match"] = 0.5  # Base score
            if "tags" in place and "cuisine" in place["tags"]:
                cuisine = place["tags"]["cuisine"]

                # For bars near Oktoberfest, favor bavarian/german places
                if context.get("nearby_landmark") == "oktoberfest" and cuisine in [
                    "bavarian",
                    "german",
                ]:
                    scores["cuisine_match"] = 0.9
                    place["cuisine_match"] = "bavarian"

                # For Isar places, favor places with outdoor seating
                if (
                    context.get("nearby_landmark") == "isar"
                    and place.get("tags", {}).get("outdoor_seating") == "yes"
                ):
                    scores["cuisine_match"] += 0.3
                    place["outdoor_seating"] = True

            # 5. Ratings score (if available)
            scores["ratings"] = 0.5  # Default middle score
            if "tags" in place and any(
                tag in place["tags"] for tag in ["stars", "rating"]
            ):
                rating = place["tags"].get("stars", place["tags"].get("rating", 0))
                try:
                    rating_val = float(rating)
                    # Normalize to 0-1 range (assuming ratings are typically 1-5)
                    scores["ratings"] = min(1.0, rating_val / 5.0)
                except (ValueError, TypeError):
                    pass  # Keep default if conversion fails

            # Calculate the final weighted score
            final_score = sum(
                score * weights[category] for category, score in scores.items()
            )

            # Store the score and component scores in the place dict
            place["relevance_score"] = final_score
            place["component_scores"] = scores

        # Sort places by final score in descending order
        ranked_places = sorted(
            places, key=lambda x: x.get("relevance_score", 0), reverse=True
        )

        return ranked_places

    def _explain_relevance(
        self, place: Dict[str, Any], context: Dict[str, Any]
    ) -> Dict[str, str]:
        """Generate human-readable explanations for why a place was selected.

        Args:
            place: The place dictionary
            context: Search context

        Returns:
            Dictionary of relevance factors
        """
        factors = {}
        scores = place.get("component_scores", {})

        # Distance factor
        if "distance" in scores and scores["distance"] > 0.7:
            factors["proximity"] = "Very close to your location"
        elif "distance" in scores and scores["distance"] > 0.4:
            factors["proximity"] = "Reasonably close to your location"

        # Landmark factor
        if "landmark_proximity" in scores and scores["landmark_proximity"] > 0.7:
            if context.get("nearby_landmark") == "isar":
                factors["location"] = "Close to the Isar river"
            elif context.get("nearby_landmark") == "oktoberfest":
                factors["location"] = "Near Oktoberfest/Theresienwiese"
            else:
                factors["location"] = "Close to key landmark"

        # Local popularity
        if "local_match" in place:
            if place["local_match"] in [
                "augustiner",
                "paulaner",
                "hofbräu",
                "löwenbräu",
            ]:
                factors["popularity"] = "Popular traditional brewery"
            elif place["local_match"] in ["biergarten", "brauhaus"]:
                factors["popularity"] = "Traditional Bavarian venue"
            else:
                factors["popularity"] = "Popular with locals"

        # Cuisine match
        if "cuisine_match" in place and place["cuisine_match"] == "bavarian":
            factors["cuisine"] = "Authentic Bavarian cuisine"

        # Outdoor seating
        if place.get("outdoor_seating"):
            factors["features"] = "Has outdoor seating"

        return factors

    def _get_osm_tags_for_category(self, category: str) -> Dict[str, List[str]]:
        """Map natural language category to OSM tags.

        Args:
            category: Natural language category

        Returns:
            Dictionary of OSM tags
        """
        # Try direct match
        if category.lower() in self.category_mappings:
            return self.category_mappings[category.lower()]

        # Try partial match
        for key, tags in self.category_mappings.items():
            if key in category.lower() or category.lower() in key:
                return tags

        return {}

    def _infer_osm_tags(self, category: str) -> Dict[str, List[str]]:
        """Infer OSM tags from a category string that doesn't match predefined mappings.

        Args:
            category: Natural language category

        Returns:
            Dictionary of OSM tags
        """
        # Start with empty tags
        inferred_tags = {}

        # Common prefixes and their mappings
        prefixes = {
            "restaurant": {"amenity": ["restaurant"]},
            "cafe": {"amenity": ["cafe"]},
            "bar": {"amenity": ["bar"]},
            "hotel": {"tourism": ["hotel"]},
            "shop": {"shop": ["convenience"]},
            "store": {"shop": ["convenience"]},
            "museum": {"tourism": ["museum"]},
            "park": {"leisure": ["park"]},
            "hospital": {"amenity": ["hospital"]},
            "school": {"amenity": ["school"]},
        }

        # Check if any prefix is in the category
        for prefix, tags in prefixes.items():
            if prefix in category.lower():
                inferred_tags.update(tags)
                break

        # If nothing found, return a generic amenity tag
        if not inferred_tags:
            inferred_tags = {"amenity": ["restaurant", "cafe", "bar"]}

        return inferred_tags

    def _search_with_overpass(
        self,
        lat: float,
        lon: float,
        radius: int,
        osm_tags: Dict[str, List[str]],
        limit: int,
        extras: Optional[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Search for places using the Overpass API.

        Args:
            lat: Latitude
            lon: Longitude
            radius: Search radius in meters
            osm_tags: OSM tags to filter by
            limit: Maximum number of results
            extras: Extra parameters for filtering

        Returns:
            List of places
        """
        # Construct Overpass query
        query_parts = []

        for tag_key, tag_values in osm_tags.items():
            for value in tag_values:
                query_parts.append(
                    f'node["{tag_key}"="{value}"](around:{radius},{lat},{lon});'
                )

        # Add extra filters if provided
        if extras:
            for key, value in extras.items():
                if isinstance(value, list):
                    for v in value:
                        query_parts.append(
                            f'node["{key}"="{v}"](around:{radius},{lat},{lon});'
                        )
                else:
                    query_parts.append(
                        f'node["{key}"="{value}"](around:{radius},{lat},{lon});'
                    )

        # Complete query
        overpass_query = f"""
        [out:json];
        (
          {" ".join(query_parts)}
        );
        out body center {limit};
        """

        logger.debug(f"Overpass query: {overpass_query}")

        # Make request to Overpass API
        response = requests.post(
            OVERPASS_API_ENDPOINT,
            data={"data": overpass_query},
            headers={"Content-Type": "application/x-www-form-urlencoded"},
        )

        if response.status_code != 200:
            logger.error(
                f"Overpass API error: {response.status_code} - {response.text}"
            )
            return []

        # Parse response
        data = response.json()
        results = []

        for element in data.get("elements", []):
            if element.get("type") != "node":
                continue

            tags = element.get("tags", {})
            name = tags.get("name", "Unnamed Place")

            # Skip places without names
            if name == "Unnamed Place":
                continue

            place_type = self._determine_place_type(tags)

            place = {
                "name": name,
                "coordinates": (element.get("lat"), element.get("lon")),
                "category": place_type,
                "address": self._format_address(tags),
                "tags": tags,
            }

            results.append(place)

        logger.info(f"Found {len(results)} places with Overpass API")
        return results

    def _determine_place_type(self, tags: Dict[str, str]) -> str:
        """Determine the place type from OSM tags.

        Args:
            tags: OSM tags

        Returns:
            Place type
        """
        if "amenity" in tags:
            return tags["amenity"].replace("_", " ").title()
        elif "shop" in tags:
            return tags["shop"].replace("_", " ").title()
        elif "tourism" in tags:
            return tags["tourism"].replace("_", " ").title()
        elif "leisure" in tags:
            return tags["leisure"].replace("_", " ").title()
        else:
            return "Point of Interest"

    def _format_address(self, tags: Dict[str, str]) -> str:
        """Format address from OSM tags.

        Args:
            tags: OSM tags

        Returns:
            Formatted address
        """
        address_parts = []

        # Street address
        if "addr:street" in tags and "addr:housenumber" in tags:
            address_parts.append(f"{tags['addr:street']} {tags['addr:housenumber']}")
        elif "addr:street" in tags:
            address_parts.append(tags["addr:street"])

        # Postal code and city
        if "addr:postcode" in tags and "addr:city" in tags:
            address_parts.append(f"{tags['addr:postcode']} {tags['addr:city']}")
        elif "addr:city" in tags:
            address_parts.append(tags["addr:city"])

        if not address_parts and "address" in tags:
            return tags["address"]

        return ", ".join(address_parts) or "Address unknown"

    def _calculate_distance(
        self, lat1: float, lon1: float, lat2: float, lon2: float
    ) -> float:
        """Calculate distance between two sets of coordinates in meters.

        Args:
            lat1: Latitude of first point
            lon1: Longitude of first point
            lat2: Latitude of second point
            lon2: Longitude of second point

        Returns:
            Distance in meters
        """
        # Approximate radius of Earth in meters
        R = 6371000

        # Convert to radians
        lat1_rad = math.radians(lat1)
        lon1_rad = math.radians(lon1)
        lat2_rad = math.radians(lat2)
        lon2_rad = math.radians(lon2)

        # Differences
        dLat = lat2_rad - lat1_rad
        dLon = lon2_rad - lon1_rad

        # Haversine formula
        a = math.sin(dLat / 2) * math.sin(dLat / 2) + math.cos(lat1_rad) * math.cos(
            lat2_rad
        ) * math.sin(dLon / 2) * math.sin(dLon / 2)
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
        distance = R * c

        return round(distance)

    def _generate_backup_places(
        self,
        category: str,
        lat: float,
        lon: float,
        radius: int,
        limit: int,
        context: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """Generate backup places when Overpass API fails.

        Args:
            category: Category of places to generate
            lat: Latitude of reference location
            lon: Longitude of reference location
            radius: Search radius
            limit: Number of places to generate
            context: Contextual information

        Returns:
            List of generated places
        """
        logger.info(f"Generating backup places for {category} near ({lat}, {lon})")

        # Normalize category
        category = category.lower()

        # Base dictionary of general places by category in Munich
        munich_places = {
            "restaurant": [
                {
                    "name": "Hofbräuhaus",
                    "coordinates": (48.1375, 11.5796),
                    "address": "Platzl 9, 80331 München",
                },
                {
                    "name": "Augustiner Keller",
                    "coordinates": (48.1406, 11.5608),
                    "address": "Arnulfstraße 52, 80335 München",
                },
                {
                    "name": "Wirtshaus in der Au",
                    "coordinates": (48.1259, 11.5843),
                    "address": "Lilienstraße 51, 81669 München",
                },
                {
                    "name": "Osterwaldgarten",
                    "coordinates": (48.1581, 11.5845),
                    "address": "Keferstraße 6, 80802 München",
                },
                {
                    "name": "Ratskeller München",
                    "coordinates": (48.1367, 11.5764),
                    "address": "Marienplatz 8, 80331 München",
                },
                {
                    "name": "Paulaner am Nockherberg",
                    "coordinates": (48.1225, 11.5811),
                    "address": "Hochstraße 77, 81541 München",
                },
                {
                    "name": "Tantris",
                    "coordinates": (48.1712, 11.5861),
                    "address": "Johann-Fichte-Straße 7, 80805 München",
                },
            ],
            "cafe": [
                {
                    "name": "Café Luitpold",
                    "coordinates": (48.1422, 11.5752),
                    "address": "Brienner Str. 11, 80333 München",
                },
                {
                    "name": "Café Glockenspiel",
                    "coordinates": (48.1371, 11.5756),
                    "address": "Marienplatz 28, 80331 München",
                },
                {
                    "name": "Man vs. Machine",
                    "coordinates": (48.1296, 11.5787),
                    "address": "Müllerstraße 23, 80469 München",
                },
                {
                    "name": "Bald Neu",
                    "coordinates": (48.1513, 11.5609),
                    "address": "Theresienstraße 72, 80333 München",
                },
                {
                    "name": "Aroma Kaffeebar",
                    "coordinates": (48.1322, 11.5728),
                    "address": "Pestalozzistraße 24, 80469 München",
                },
            ],
            "bar": [
                {
                    "name": "Schumann's Bar",
                    "coordinates": (48.1462, 11.5779),
                    "address": "Odeonsplatz 6-7, 80539 München",
                },
                {
                    "name": "Zephyr Bar",
                    "coordinates": (48.1288, 11.5769),
                    "address": "Baaderstraße 68, 80469 München",
                },
                {
                    "name": "Ory Bar",
                    "coordinates": (48.1422, 11.5770),
                    "address": "Mandarin Oriental, Neuturmstraße 1, 80331 München",
                },
                {
                    "name": "Pusser Bar",
                    "coordinates": (48.1430, 11.5832),
                    "address": "Falkenturmstraße 9, 80331 München",
                },
                {
                    "name": "Goldene Bar",
                    "coordinates": (48.1437, 11.5901),
                    "address": "Prinzregentenstraße 1, 80538 München",
                },
            ],
            "supermarket": [
                {
                    "name": "REWE",
                    "coordinates": (48.1400, 11.5618),
                    "address": "Karlsplatz 5, 80335 München",
                },
                {
                    "name": "Edeka",
                    "coordinates": (48.1330, 11.5740),
                    "address": "Fraunhoferstraße 13, 80469 München",
                },
                {
                    "name": "Lidl",
                    "coordinates": (48.1458, 11.5564),
                    "address": "Ludwigsvorstadt-Isarvorstadt, 80336 München",
                },
                {
                    "name": "Aldi Süd",
                    "coordinates": (48.1383, 11.5700),
                    "address": "Sonnenstraße 23, 80331 München",
                },
            ],
            "museum": [
                {
                    "name": "Deutsches Museum",
                    "coordinates": (48.1299, 11.5833),
                    "address": "Museumsinsel 1, 80538 München",
                },
                {
                    "name": "Pinakothek der Moderne",
                    "coordinates": (48.1474, 11.5724),
                    "address": "Barer Str. 40, 80333 München",
                },
                {
                    "name": "Alte Pinakothek",
                    "coordinates": (48.1488, 11.5700),
                    "address": "Barer Str. 27, 80333 München",
                },
                {
                    "name": "Museum Brandhorst",
                    "coordinates": (48.1477, 11.5742),
                    "address": "Theresienstraße 35a, 80333 München",
                },
            ],
            "hotel": [
                {
                    "name": "Hotel Bayerischer Hof",
                    "coordinates": (48.1404, 11.5759),
                    "address": "Promenadeplatz 2-6, 80333 München",
                },
                {
                    "name": "Mandarin Oriental Munich",
                    "coordinates": (48.1408, 11.5755),
                    "address": "Neuturmstraße 1, 80331 München",
                },
                {
                    "name": "Rocco Forte The Charles Hotel",
                    "coordinates": (48.1416, 11.5651),
                    "address": "Sophienstraße 28, 80333 München",
                },
                {
                    "name": "Eden Hotel Wolff",
                    "coordinates": (48.1421, 11.5584),
                    "address": "Arnulfstraße 4, 80335 München",
                },
                {
                    "name": "King's Hotel Center",
                    "coordinates": (48.1425, 11.5600),
                    "address": "Marsstraße 15, 80335 München",
                },
            ],
            "bakery": [
                {
                    "name": "Rischart",
                    "coordinates": (48.1371, 11.5754),
                    "address": "Marienplatz 18, 80331 München",
                },
                {
                    "name": "Zeit für Brot",
                    "coordinates": (48.1310, 11.5760),
                    "address": "Frauenstraße 11, 80469 München",
                },
                {
                    "name": "Zöttl",
                    "coordinates": (48.1526, 11.5785),
                    "address": "Mönchstraße 13, 80805 München",
                },
                {
                    "name": "Bäckerei Wimmer",
                    "coordinates": (48.1348, 11.5791),
                    "address": "Steinstraße 27, 81667 München",
                },
            ],
        }

        # Check if we have places for the requested category
        normalized_category = None
        for key in munich_places.keys():
            if key in category or category in key:
                normalized_category = key
                break

        # Fallback to restaurant if category not found
        if not normalized_category:
            normalized_category = "restaurant"
            logger.debug(f"Using 'restaurant' as fallback for category '{category}'")

        # Get places for the category
        places = munich_places[normalized_category]

        # Add distance information
        for place in places:
            place_lat, place_lon = place["coordinates"]
            # Calculate distance to reference point
            distance = self._calculate_distance(lat, lon, place_lat, place_lon)
            place["distance"] = distance

        # Sort by distance and limit results
        places.sort(key=lambda x: x["distance"])
        places = places[:limit]

        # Format places for return
        result = []
        for i, place in enumerate(places):
            result.append(
                {
                    "id": f"backup-{normalized_category}-{i}",
                    "name": place["name"],
                    "coordinates": place["coordinates"],
                    "distance": place["distance"],
                    "address": place.get("address", "Unknown address"),
                    "category": normalized_category,
                }
            )

        return result


def parse_fuzzy_location(query: str) -> Tuple[str, Optional[str], Dict[str, Any]]:
    """Parse a fuzzy location query to extract category and location.

    Args:
        query: The fuzzy location query (e.g. "good restaurant near central station")

    Returns:
        Tuple of (category, location reference, attributes)
    """
    logger.debug(f"Parsing fuzzy location query: {query}")

    # Define attribute keywords for different aspects
    attribute_keywords = {
        "quality": [
            "good",
            "best",
            "nice",
            "top",
            "great",
            "excellent",
            "recommended",
            "popular",
            "favorite",
            "quality",
        ],
        "cost": [
            "cheap",
            "expensive",
            "affordable",
            "budget",
            "luxury",
            "high-end",
            "low-cost",
            "upscale",
        ],
        "authenticity": ["authentic", "traditional", "real", "genuine", "local"],
        "speed": ["fast", "quick", "express", "rapid", "speedy"],
        "atmosphere": [
            "cozy",
            "quiet",
            "romantic",
            "family",
            "friendly",
            "casual",
            "elegant",
        ],
        "food_quality": [
            "delicious",
            "tasty",
            "fresh",
            "healthy",
            "organic",
            "gourmet",
        ],
    }

    # Add cuisine types as attributes to better handle specific cuisine requests
    cuisine_types = {
        "bavarian": [
            "bavarian",
            "german",
            "munich",
            "münchen",
            "oktoberfest",
            "biergarten",
            "beer garden",
        ],
        "italian": ["italian", "pasta", "pizza", "risotto", "mediterranean"],
        "chinese": ["chinese", "asian", "dim sum", "wok"],
        "japanese": ["japanese", "sushi", "ramen", "asian"],
        "thai": ["thai", "asian", "spicy"],
        "mexican": ["mexican", "tacos", "burritos", "latin american"],
        "indian": ["indian", "curry", "spicy"],
        "greek": ["greek", "mediterranean"],
        "french": ["french", "bistro", "gourmet"],
        "spanish": ["spanish", "tapas", "mediterranean"],
        "american": ["american", "burger", "steak", "bbq", "barbecue"],
    }

    # Flatten attributes for detection
    all_attributes = {}
    for category, terms in attribute_keywords.items():
        for term in terms:
            all_attributes[term] = category

    # Initialize attributes dictionary with counters for each attribute type
    attributes = {category: 0 for category in attribute_keywords.keys()}
    attributes["general"] = []  # For storing raw attribute strings

    # Also add entries for cuisine types
    for cuisine in cuisine_types.keys():
        attributes[cuisine] = 0

    # Clean the query
    query = query.lower().strip()
    words = query.split()

    # Find all mentioned attributes
    for word in words:
        word = word.strip(",.!?")
        if word in all_attributes:
            attr_type = all_attributes[word]
            attributes[attr_type] += 1
            attributes["general"].append(word)

    # Look for cuisine types in the query
    query_lower = query.lower()
    for cuisine, terms in cuisine_types.items():
        for term in terms:
            if term in query_lower:
                attributes[cuisine] += 1
                attributes["general"].append(
                    cuisine
                )  # Add the cuisine type to general attributes
                break  # Break after finding the first match for this cuisine

    # Patterns to identify location parts
    location_patterns = [
        r"(?:near|close to|around|by|at|in|next to)\s+(?:the\s+)?(.+?)(?:$|,|\.|and|or|\s+with|\s+that|\s+which)",
        r"(?:in|at)\s+(?:the\s+)?(.+?)(?:$|,|\.|and|or|\s+with|\s+that|\s+which)",
        r"(?:close|near|nearby|around)\s+(?:the\s+)?(.+?)(?:$|,|\.|and|or|\s+with|\s+that|\s+which)",
    ]

    # Find category
    # Common food categories
    common_food_categories = [
        "restaurant",
        "cafe",
        "bar",
        "coffee",
        "fast food",
        "supermarket",
        "bakery",
        "coffee shop",
        "bistro",
        "pub",
        "diner",
        "pizzeria",
        "italian",
        "chinese",
        "mexican",
        "thai",
        "indian",
        "japanese",
        "greek",
        "breakfast",
        "lunch",
        "dinner",
        "food",
        "eating",
        "drink",
        "meal",
        "brunch",
        "steak",
        "seafood",
        "sushi",
        "burger",
        "sandwich",
        "bavarian",  # Make sure key cuisine types are included
    ]

    # Try to find a category from common terms
    category = None
    for food_term in common_food_categories:
        if food_term in query:
            category = food_term
            break

    # Fallback to generic categories if not found
    if not category:
        # Try to extract based on pattern "find a X" or "looking for a X"
        category_patterns = [
            r"(?:find|looking\s+for|want|need|searching\s+for|seeking|get)\s+(?:a|an|some|the)?\s+(.+?)(?:\s+near|\s+close|\s+in|\s+at|\s+by|$|,|\.|and|or)",
            r"(?:a|an|some|the)\s+(.+?)(?:\s+near|\s+close|\s+in|\s+at|\s+by|$|,|\.|and|or)",
        ]

        for pattern in category_patterns:
            match = re.search(pattern, query)
            if match:
                potential_category = match.group(1).strip()
                # Filter out very short or common words
                if len(potential_category) > 2 and potential_category not in [
                    "place",
                    "spot",
                    "location",
                ]:
                    category = potential_category
                    break

        # Last resort: use the first noun in the query
        if not category:
            # Default to "restaurant" as a common catch-all
            category = "restaurant"

    # Find location reference
    location = None
    for pattern in location_patterns:
        match = re.search(pattern, query)
        if match:
            location = match.group(1).strip()
            break

    # Check if we detected a specific cuisine but didn't set it as category
    if category == "restaurant":
        # If we detected a specific cuisine type with high confidence, update the category
        for cuisine in cuisine_types:
            if attributes.get(cuisine, 0) > 0:
                category = f"{cuisine} {category}"
                break

    logger.debug(
        f"Parsed query: category={category}, location={location}, attributes={attributes}"
    )
    return category, location, attributes
