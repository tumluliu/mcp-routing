"""LLM interface for DeepSeek."""

import json
import requests
import base64
from typing import Dict, List, Optional, Any, Union, Tuple
import os
import traceback
from loguru import logger
import re
import time
from PIL import Image
import io
from pathlib import Path

from .config import (
    DEEPSEEK_ENDPOINT,
    DEEPSEEK_API_KEY,
    DEEPSEEK_MODEL,
    MUNICH_CENTER,
    DEFAULT_SEARCH_RADIUS,
)
from .places import PlaceSearchService, parse_fuzzy_location


class DeepSeekLLM:
    """Interface for DeepSeek language model."""

    def __init__(
        self,
        model: str = DEEPSEEK_MODEL,
        endpoint: str = DEEPSEEK_ENDPOINT,
        api_key: str = DEEPSEEK_API_KEY,
    ):
        """Initialize the DeepSeek interface.

        Args:
            model: The DeepSeek model to use
            endpoint: The API endpoint for DeepSeek
            api_key: API key for DeepSeek
        """
        self.model = model
        self.endpoint = endpoint
        self.api_key = api_key
        self.headers = {"Content-Type": "application/json"}
        if api_key:
            self.headers["Authorization"] = f"Bearer {api_key}"
        else:
            logger.warning("No API key provided for DeepSeek LLM")

        # Initialize conversation contexts storage
        self.conversations = {}

        # Initialize place search service
        self.place_search_service = PlaceSearchService()

        logger.info(
            f"Initialized DeepSeekLLM with model: {model}, endpoint: {endpoint}"
        )

    def _call_api(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.0,
        stream: bool = False,
    ) -> Dict[str, Any]:
        """Make a raw API call to DeepSeek.

        Args:
            messages: List of message dictionaries
            temperature: Sampling temperature
            stream: Whether to stream the response

        Returns:
            API response
        """
        try:
            endpoint_url = f"{self.endpoint}/chat/completions"
            payload = {
                "model": self.model,
                "messages": messages,
                "temperature": temperature,
                "stream": stream,
            }

            # Log request details (but censor the API key)
            safe_headers = self.headers.copy()
            if "Authorization" in safe_headers:
                safe_headers["Authorization"] = "Bearer [REDACTED]"

            logger.debug(f"DeepSeek API request to {endpoint_url}")
            logger.debug(f"Headers: {safe_headers}")

            # Don't log the entire messages array as it can be very large
            messages_summary = f"{len(messages)} messages"
            if len(messages) > 0:
                messages_summary += (
                    f" (system: {messages[0].get('content', '')[:50]}...)"
                )

            logger.debug(
                f"Payload: {{\n  'model': '{self.model}',\n  'messages': {messages_summary},\n  'temperature': {temperature},\n  'stream': {stream}\n}}"
            )

            if stream:
                logger.debug("Making streaming request to DeepSeek API")
                response = requests.post(
                    endpoint_url,
                    headers=self.headers,
                    json=payload,
                    stream=True,
                )

                if response.status_code != 200:
                    error_msg = (
                        f"DeepSeek API error: {response.status_code} - {response.text}"
                    )
                    logger.error(error_msg)
                    raise Exception(error_msg)

                logger.debug("Streaming response started from DeepSeek API")
                return response
            else:
                logger.debug("Making non-streaming request to DeepSeek API")
                response = requests.post(
                    endpoint_url,
                    headers=self.headers,
                    json=payload,
                )

                if response.status_code != 200:
                    error_msg = (
                        f"DeepSeek API error: {response.status_code} - {response.text}"
                    )
                    logger.error(error_msg)
                    raise Exception(error_msg)

                logger.debug(
                    f"DeepSeek API response received, length: {len(response.text)} bytes"
                )
                return response.json()

        except requests.exceptions.RequestException as e:
            logger.error(f"Network error calling DeepSeek API: {str(e)}")
            logger.error(traceback.format_exc())
            raise Exception(f"Network error calling DeepSeek API: {str(e)}")
        except Exception as e:
            logger.error(f"Error calling DeepSeek API: {str(e)}")
            logger.error(traceback.format_exc())
            raise

    def _prepare_image_for_message(self, image_path: str) -> Dict[str, Any]:
        """Prepare an image to be included in a message.

        Args:
            image_path: Path to the image file

        Returns:
            Image content object to be included in a message
        """
        if not image_path or not os.path.exists(image_path):
            logger.warning(f"Image file not found or invalid path: {image_path}")
            return None

        try:
            logger.debug(f"Processing image for LLM: {image_path}")
            # Read image file as base64
            with open(image_path, "rb") as image_file:
                base64_image = base64.b64encode(image_file.read()).decode("utf-8")

            logger.debug(f"Image encoded as base64, size: {len(base64_image)} bytes")

            # Create image content object
            # Note: The exact format might vary based on DeepSeek's multimodal API specifics
            # This is based on the OpenAI format
            return {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
            }
        except Exception as e:
            logger.error(f"Error processing image: {str(e)}")
            logger.error(traceback.format_exc())
            return None

    def create_conversation(self, conversation_id: Optional[str] = None) -> str:
        """Create a new conversation or initialize an existing one.

        Args:
            conversation_id: Optional ID for the conversation

        Returns:
            The conversation ID
        """
        import uuid

        # Generate a random ID if none provided
        if not conversation_id:
            conversation_id = str(uuid.uuid4())
            logger.debug(f"Generated new conversation ID: {conversation_id}")
        else:
            logger.debug(f"Using provided conversation ID: {conversation_id}")

        # Initialize conversation with empty history if it doesn't exist
        if conversation_id not in self.conversations:
            logger.debug(f"Initializing new conversation: {conversation_id}")
            self.conversations[conversation_id] = []
        else:
            logger.debug(
                f"Using existing conversation with {len(self.conversations[conversation_id])} messages: {conversation_id}"
            )

        return conversation_id

    def parse_routing_query(
        self,
        query: str,
        conversation_id: Optional[str] = None,
        stream_response: bool = False,
        image_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Parse a natural language routing query into structured parameters.

        Args:
            query: The natural language query
            conversation_id: Optional conversation ID for context
            stream_response: Whether to stream the response
            image_path: Optional path to an image to include with the query

        Returns:
            Dictionary of structured routing parameters
        """
        logger.info(f"Parsing routing query: {query}")
        if image_path:
            logger.info(f"Query includes image: {image_path}")

        try:
            # Check if the query contains fuzzy location references
            if self._is_fuzzy_location_query(query):
                logger.info("Detected fuzzy location query")
                return self._handle_fuzzy_location_query(query, conversation_id)

            # Ensure conversation exists
            if conversation_id:
                conversation_id = self.create_conversation(conversation_id)
                conversation_history = self.conversations.get(conversation_id, [])
                logger.debug(
                    f"Using conversation history with {len(conversation_history)} messages"
                )
            else:
                conversation_history = []
                logger.debug("No conversation history (new conversation)")

            system_prompt = """
            You are a helpful assistant that converts natural language routing queries into structured parameters.
            Extract the origin, destination, and any other relevant parameters from the query.
            For addresses in Munich, return detailed information.
            
            If an image is provided, analyze it to extract relevant landmark information.
            The image could contain places like Marienplatz, the English Garden, Olympiapark, BMW Museum, 
            Deutsches Museum, Nymphenburg Palace, or other landmarks in Munich.
            
            IMPORTANT: You MUST return your response ONLY as a valid JSON object with these fields:
            - origin: Origin location (address or coordinates)
            - destination: Destination location (address or coordinates)
            - mode: Transportation mode (default: "driving")
            - waypoints: List of intermediate waypoints (optional)
            - avoid: List of features to avoid (optional)
            - departure_time: Departure time (optional)
            - arrival_time: Arrival time (optional)
            - landmarks: Information about landmarks identified in the image (optional)
            
            DO NOT provide explanations, details, or any other text outside of the JSON object.
            DO NOT use markdown formatting. ONLY return a valid JSON object.
            
            All addresses should be formatted properly for Munich, Germany.
            
            If the user references locations from previous conversations (like "there", "that place", 
            "the same location", etc.), use the context from previous messages to determine the actual locations.
            
            If an image is included but no text query is provided, extract as much information as possible from the
            image and respond with information about the landmark and how it can be used for routing, still in JSON format.
            """

            # Construct messages with conversation history and current query
            messages = [{"role": "system", "content": system_prompt}]

            # Add conversation history
            messages.extend(conversation_history)

            # Prepare user message
            user_message = {"role": "user"}

            # If there's an image, add it to the content
            if image_path:
                logger.debug("Processing image for routing query")
                # For multimodal models, the content should be an array
                content = []

                # Add text if available
                if query and query.strip():
                    content.append({"type": "text", "text": query})
                    logger.debug(f"Added text content: {query}")

                # Process and add image
                image_content = self._prepare_image_for_message(image_path)
                if image_content:
                    content.append(image_content)
                    logger.debug("Added image content to message")
                else:
                    logger.warning("Failed to process image for message")

                if not content:
                    # Fallback if no content could be added
                    fallback_text = "Please analyze this image for landmarks in Munich."
                    content = [{"type": "text", "text": fallback_text}]
                    logger.debug(f"Using fallback text content: {fallback_text}")

                user_message["content"] = content
            else:
                # Simple text message
                user_message["content"] = query
                logger.debug(f"Using simple text message: {query}")

            # Add user message
            messages.append(user_message)
            logger.debug(f"Final message count: {len(messages)}")

            if stream_response:
                logger.debug("Using streaming response for routing query parsing")
                # Return the streaming response for the caller to process
                return {
                    "stream": True,
                    "response": self._call_api(messages, stream=True),
                    "conversation_id": conversation_id,
                    "messages": messages,
                    "query_type": "routing",
                }

            logger.debug("Making non-streaming API call for routing query parsing")
            response = self._call_api(messages)
            response_text = response["choices"][0]["message"]["content"]
            logger.debug(f"Received response length: {len(response_text)} characters")

            # Update conversation history if using a conversation context
            if conversation_id:
                logger.debug(f"Updating conversation history for {conversation_id}")
                self.conversations[conversation_id].append(user_message)
                self.conversations[conversation_id].append(
                    {"role": "assistant", "content": response_text}
                )

            # Three-stage parsing attempt
            result = None

            # Stage 1: Try to parse as JSON directly
            try:
                # Extract JSON from response
                json_str = response_text
                if "```json" in response_text:
                    logger.debug("Extracting JSON from markdown code block (json)")
                    json_str = response_text.split("```json")[1].split("```")[0].strip()
                elif "```" in response_text:
                    logger.debug("Extracting JSON from markdown code block (generic)")
                    json_str = response_text.split("```")[1].split("```")[0].strip()
                else:
                    logger.debug("Using full response as JSON string")

                logger.debug(f"Parsing JSON string: {json_str[:100]}...")
                result = json.loads(json_str)
                logger.info(f"Successfully parsed routing query: {result}")
                return result
            except (json.JSONDecodeError, IndexError) as e:
                logger.error(f"Failed to parse LLM response as JSON: {str(e)}")
                logger.debug(f"Response text: {response_text}")

            # Stage 2: Try to extract structured data from text
            if result is None:
                try:
                    logger.info(
                        "Attempting to extract structured information from text response"
                    )
                    result = self._extract_routing_info_from_text(response_text)
                    if result:
                        logger.info(
                            f"Successfully extracted routing info: {list(result.keys())}"
                        )
                        return result
                except Exception as extraction_error:
                    logger.error(
                        f"Failed to extract routing info from text: {str(extraction_error)}"
                    )

            # Stage 3: Make a second attempt with explicit JSON instructions
            if result is None:
                try:
                    logger.warning(
                        "Making second attempt with explicit JSON instructions"
                    )
                    clarification_message = {
                        "role": "user",
                        "content": """I need the routing data in structured JSON format only. 
                        Please provide ONLY a valid JSON object with these fields:
                        - origin: Origin location
                        - destination: Destination location
                        - mode: Transportation mode

                        NO explanations, NO markdown formatting, ONLY valid JSON.""",
                    }
                    messages.append(clarification_message)

                    clarification_response = self._call_api(messages)
                    clarification_text = clarification_response["choices"][0][
                        "message"
                    ]["content"]

                    # Try to extract JSON from the clarification response
                    if "```json" in clarification_text:
                        json_str = (
                            clarification_text.split("```json")[1]
                            .split("```")[0]
                            .strip()
                        )
                    elif "```" in clarification_text:
                        json_str = (
                            clarification_text.split("```")[1].split("```")[0].strip()
                        )
                    else:
                        json_str = clarification_text

                    result = json.loads(json_str)
                    logger.info(
                        f"Successfully parsed routing query from second attempt: {list(result.keys())}"
                    )
                    return result
                except Exception as second_error:
                    logger.error(f"Second parsing attempt failed: {str(second_error)}")

            # Stage 4: Use fallback as a last resort
            logger.warning("All parsing attempts failed, using fallback values")
            return self._create_fallback_routing_result(query, response_text)

        except Exception as e:
            logger.error(f"Error during routing query parsing: {str(e)}")
            logger.error(traceback.format_exc())

            # Even if everything fails, still provide a fallback result
            try:
                return self._create_fallback_routing_result(
                    query, "Error occurred during processing"
                )
            except:
                # Absolute last resort
                return {
                    "origin": "Marienplatz, Munich",
                    "destination": "Munich Hauptbahnhof",
                    "mode": "driving",
                    "_fallback": True,
                    "_error": f"Critical error: {str(e)}",
                    "_original_query": query,
                }

    def _extract_routing_info_from_text(self, text: str) -> Dict[str, Any]:
        """Extract routing information from text response.

        Args:
            text: Text response from LLM

        Returns:
            Dictionary with extracted routing parameters
        """
        # Default values
        result = {
            "origin": None,
            "destination": None,
            "mode": "driving",
        }

        # Check for detailed route keyword patterns which are common in LLM responses
        if (
            "route from" in text.lower()
            or "driving route" in text.lower()
            or "directions from" in text.lower()
        ):
            logger.debug("Detected route description pattern in text")

        # Look for origin and destination in the markdown headings or titles
        if "from" in text.lower() and "to" in text.lower():
            title_pattern = re.search(
                r"(?i)(route|directions|path|way)\s+from\s+[\"']?([^\"']+)[\"']?\s+to\s+[\"']?([^\"']+)[\"']?",
                text,
            )
            if title_pattern:
                result["origin"] = title_pattern.group(2).strip()
                result["destination"] = title_pattern.group(3).strip()
                logger.debug(
                    f"Extracted origin/destination from title pattern: {result['origin']} to {result['destination']}"
                )

        # Look for markdown-formatted bold content
        bold_patterns = re.findall(r"\*\*([^*]+)\*\*", text)
        if bold_patterns:
            logger.debug(f"Found {len(bold_patterns)} bold-formatted sections")

            # Look for patterns that might indicate locations (specific to route descriptions)
            for idx, bold_text in enumerate(bold_patterns):
                bold_text = bold_text.strip()

                # Check for specific locations that are likely to be highlighted
                if any(
                    loc in bold_text.lower()
                    for loc in [
                        "airport",
                        "hauptbahnhof",
                        "marienplatz",
                        "station",
                        "terminal",
                    ]
                ):
                    logger.debug(f"Found potential landmark in bold text: {bold_text}")
                    if not result["origin"]:
                        result["origin"] = bold_text
                    elif not result["destination"]:
                        result["destination"] = bold_text

        # Look for origin and destination in the text lines
        lines = text.split("\n")

        # Extract locations from markdown headers or bold text
        for line in lines:
            line = line.strip()

            # Look for explicit origin/destination markers
            if re.search(r"(?i)(?:from|origin|start|depart|departure)[\s:]", line):
                potential_origin = self._extract_location_from_line(line)
                if potential_origin and not result["origin"]:
                    result["origin"] = potential_origin
                    logger.debug(f"Extracted origin from line: {result['origin']}")

            # Look for destination/to
            if re.search(r"(?i)(?:to|destination|arrive|arrival)[\s:]", line):
                potential_dest = self._extract_location_from_line(line)
                if potential_dest and not result["destination"]:
                    result["destination"] = potential_dest
                    logger.debug(
                        f"Extracted destination from line: {result['destination']}"
                    )

            # Look for transportation mode indicators
            if "by car" in line.lower() or "driving" in line.lower():
                result["mode"] = "driving"
            elif "by bike" in line.lower() or "cycling" in line.lower():
                result["mode"] = "cycling"
            elif "on foot" in line.lower() or "walking" in line.lower():
                result["mode"] = "walking"

        # Special handling for Munich Airport in the response
        if any(
            airport_term in text.lower()
            for airport_term in [
                "munich airport",
                "flughafen münchen",
                "flughafen munich",
                "muc",
            ]
        ):
            if not result["origin"] and "depart" in text.lower():
                result["origin"] = "Munich Airport"
                logger.debug("Set origin to Munich Airport based on context")
            elif not result["destination"] and "arriv" in text.lower():
                result["destination"] = "Munich Airport"
                logger.debug("Set destination to Munich Airport based on context")

        # Special handling for main train station in the response
        if any(
            station_term in text.lower()
            for station_term in [
                "hauptbahnhof",
                "main station",
                "central station",
                "main train station",
            ]
        ):
            if not result["origin"] and "depart" in text.lower():
                result["origin"] = "Munich Hauptbahnhof"
                logger.debug("Set origin to Munich Hauptbahnhof based on context")
            elif not result["destination"] and "arriv" in text.lower():
                result["destination"] = "Munich Hauptbahnhof"
                logger.debug("Set destination to Munich Hauptbahnhof based on context")

        # If we have both origin and destination, return the result
        if result["origin"] and result["destination"]:
            return result

        # If we still don't have both origin and destination, try one more approach:
        # Extract locations from section headers (### Title)
        header_matches = re.findall(r"###\s+[^#\n]+", text)
        if header_matches:
            for header in header_matches[:2]:  # Consider first two headers only
                header_text = header.replace("###", "").strip()
                if "depart" in header_text.lower() or "start" in header_text.lower():
                    location = re.search(r"(?:from|at)\s+([^()]+)", header_text)
                    if location and not result["origin"]:
                        result["origin"] = location.group(1).strip()
                elif (
                    "arrive" in header_text.lower()
                    or "destination" in header_text.lower()
                ):
                    location = re.search(r"(?:to|at)\s+([^()]+)", header_text)
                    if location and not result["destination"]:
                        result["destination"] = location.group(1).strip()

        # Final check - if we found only one location, make educated guess on the other
        if result["origin"] and not result["destination"]:
            # Guess a common destination if we have only origin
            result["destination"] = "Munich Hauptbahnhof"
            logger.debug("Using Munich Hauptbahnhof as default destination")
        elif result["destination"] and not result["origin"]:
            # Guess a common origin if we have only destination
            result["origin"] = "Marienplatz, Munich"
            logger.debug("Using Marienplatz as default origin")

        # If we still don't have both, return None
        if not result["origin"] or not result["destination"]:
            return None

        return result

    def _extract_location_from_line(self, line: str) -> Optional[str]:
        """Extract location from a line of text.

        Args:
            line: Line of text

        Returns:
            Extracted location or None
        """
        # Try to extract from bold text
        if "**" in line:
            bold_parts = line.split("**")
            for i in range(1, len(bold_parts), 2):  # Get only the bold parts
                if bold_parts[i].strip():
                    return bold_parts[i].strip()

        # Try to extract from patterns like "from: Location" or "to: Location"
        if ":" in line:
            parts = line.split(":", 1)
            if len(parts) > 1 and parts[1].strip():
                return parts[1].strip()

        # Try to extract from quotation marks
        if '"' in line:
            parts = line.split('"')
            for i in range(1, len(parts), 2):  # Get quoted parts
                if parts[i].strip():
                    return parts[i].strip()

        # Fallback to extracting anything after common prepositions
        prepositions = ["from", "to", "at", "in", "near"]
        for prep in prepositions:
            if f" {prep} " in line.lower():
                parts = line.lower().split(f" {prep} ", 1)
                if len(parts) > 1 and parts[1].strip():
                    # Get the first few words after the preposition
                    words = parts[1].strip().split()
                    location = " ".join(words[: min(5, len(words))])
                    # Remove punctuation at the end
                    return location.rstrip(",.;:")

        return None

    def _create_fallback_routing_result(
        self, query: str, response_text: str
    ) -> Dict[str, Any]:
        """Create a fallback routing result when parsing fails.

        Args:
            query: The original query
            response_text: The LLM response text

        Returns:
            Fallback routing parameters
        """
        # Try to identify any locations in the original query
        logger.warning(f"Creating fallback routing result for query: {query}")

        # Default to common locations in Munich
        result = {
            "origin": "Marienplatz, Munich",
            "destination": "Munich Hauptbahnhof",
            "mode": "driving",
            "_fallback": True,
            "_error": "Failed to parse LLM response",
            "_original_query": query,
        }

        # If "airport" is mentioned, use it as destination
        if "airport" in query.lower() or "flughafen" in query.lower():
            result["destination"] = "Munich Airport"

        # If "hauptbahnhof" or "main station" is mentioned, use it
        if "hauptbahnhof" in query.lower() or "main station" in query.lower():
            result["origin"] = "Munich Hauptbahnhof"

        return result

    def generate_navigation_instructions(
        self,
        route_data: Dict[str, Any],
        conversation_id: Optional[str] = None,
        stream_response: bool = False,
    ) -> List[str]:
        """Generate human-friendly navigation instructions from route data.

        Args:
            route_data: The routing data from the routing engine
            conversation_id: Optional conversation ID for context
            stream_response: Whether to stream the response

        Returns:
            List of navigation instructions
        """
        try:
            logger.info("Generating navigation instructions")
            # Ensure conversation exists
            if conversation_id:
                conversation_id = self.create_conversation(conversation_id)
                conversation_history = self.conversations.get(conversation_id, [])
                logger.debug(
                    f"Using conversation history with {len(conversation_history)} messages"
                )
            else:
                conversation_history = []
                logger.debug("No conversation history (new conversation)")

            system_prompt = """
            You are a helpful navigation assistant. Convert the technical routing data into clear, 
            step-by-step navigation instructions that are easy for humans to follow while driving.
            
            Return a list of instructions, one for each major navigation step.
            Focus on important turns, street names, and landmarks.
            Use cardinal directions (North, South, East, West) when appropriate.
            Distances should be in kilometers or meters as appropriate.
            """

            # Start with system prompt
            messages = [{"role": "system", "content": system_prompt}]

            # Add conversation history
            messages.extend(conversation_history)

            # Prepare route data summary for prompt
            origin = route_data.get("origin", "Unknown")
            destination = route_data.get("destination", "Unknown")

            if isinstance(origin, (list, tuple)):
                origin_str = f"{origin[0]:.6f}, {origin[1]:.6f}"
            else:
                origin_str = str(origin)

            if isinstance(destination, (list, tuple)):
                dest_str = f"{destination[0]:.6f}, {destination[1]:.6f}"
            else:
                dest_str = str(destination)

            total_distance = route_data.get("distance", 0)
            total_duration = route_data.get("duration", 0)

            # Prepare a summary of the steps for better instructions
            steps_summary = []
            for i, step in enumerate(
                route_data.get("steps", [])[:10]
            ):  # Limit to first 10 steps
                step_str = f"{i+1}. {step.get('instruction', 'Move')} on {step.get('name', 'unnamed road')} "
                step_str += (
                    f"for {step.get('distance', 0)}m ({step.get('duration', 0)}s)"
                )
                steps_summary.append(step_str)

            all_steps_summary = "\n".join(steps_summary)
            if len(route_data.get("steps", [])) > 10:
                all_steps_summary += (
                    f"\n... and {len(route_data.get('steps', [])) - 10} more steps"
                )

            logger.debug(
                f"Prepared route summary: {origin_str} to {dest_str}, {total_distance}m, {total_duration}s"
            )

            # Prepare the user message with route information
            route_prompt = f"""
            Generate navigation instructions for this route:
            
            Origin: {origin_str}
            Destination: {dest_str}
            Distance: {total_distance} meters
            Duration: {total_duration} seconds
            
            Steps Summary:
            {all_steps_summary}
            
            Return a numbered list of clear navigation instructions.
            """

            # Add user message
            user_message = {"role": "user", "content": route_prompt}
            messages.append(user_message)

            if stream_response:
                logger.debug("Using streaming response for navigation instructions")
                return {
                    "stream": True,
                    "response": self._call_api(messages, stream=True),
                    "conversation_id": conversation_id,
                    "messages": messages,
                    "query_type": "navigation",
                }

            # Get response from LLM
            logger.debug("Making non-streaming API call for navigation instructions")
            response = self._call_api(messages)
            response_text = response["choices"][0]["message"]["content"]
            logger.debug(f"Received response length: {len(response_text)} characters")

            # Extract instructions from the response
            instructions = []
            for line in response_text.split("\n"):
                line = line.strip()
                if not line:
                    continue

                # Look for numbered lines or bullet points
                if line[0].isdigit() or line[0] in ["-", "*", "•"]:
                    # Remove the number/bullet and clean up
                    clean_line = line
                    for prefix in [".", ")", "-", "*", "•"]:
                        if prefix in clean_line:
                            parts = clean_line.split(prefix, 1)
                            if parts[0].strip().isdigit():
                                clean_line = parts[1].strip()
                                break

                    if clean_line:
                        instructions.append(clean_line)

            # If we failed to find numbered instructions, just split by lines
            if not instructions:
                logger.warning("Failed to parse numbered instructions, using raw lines")
                instructions = [
                    line.strip() for line in response_text.split("\n") if line.strip()
                ]

            # Update conversation history if using a conversation context
            if conversation_id:
                logger.debug(f"Updating conversation history for {conversation_id}")
                self.conversations[conversation_id].append(user_message)
                self.conversations[conversation_id].append(
                    {"role": "assistant", "content": response_text}
                )

            # Clean up instructions (remove any remaining bullet points or prefixes)
            clean_instructions = []
            for instr in instructions:
                # Remove any remaining numbers at the beginning
                cleaned = instr
                if cleaned and cleaned[0].isdigit():
                    cleaned = cleaned.lstrip("0123456789").strip()
                    if cleaned and cleaned[0] in [".", ")", ":", "-"]:
                        cleaned = cleaned[1:].strip()

                if cleaned:
                    clean_instructions.append(cleaned)

            logger.info(f"Generated {len(clean_instructions)} navigation instructions")
            return clean_instructions

        except Exception as e:
            logger.error(f"Error generating navigation instructions: {str(e)}")
            logger.error(traceback.format_exc())
            # Return a basic instruction set in case of error
            return [
                "Error generating detailed instructions. Please proceed with caution."
            ]

    def chat(
        self,
        query: str,
        conversation_id: Optional[str] = None,
        stream_response: bool = False,
        image_path: Optional[str] = None,
    ) -> Union[str, Dict[str, Any]]:
        """Have a free-form conversation with the LLM.

        Args:
            query: The user's message
            conversation_id: Optional conversation ID for context
            stream_response: Whether to stream the response
            image_path: Optional path to an image to include with the query

        Returns:
            The LLM's response or streaming response object
        """
        try:
            logger.info(f"Processing chat message: {query}")
            if image_path:
                logger.info(f"Chat includes image: {image_path}")

            # Ensure conversation exists
            conversation_id = self.create_conversation(conversation_id)
            conversation_history = self.conversations.get(conversation_id, [])
            logger.debug(
                f"Using conversation history with {len(conversation_history)} messages"
            )

            system_prompt = """
            You are a helpful navigation and routing assistant for Munich, Germany.
            Your primary purpose is to help users find routes and navigate the city.
            
            - If the user asks about routes, directions, or navigation, try to extract key information.
            - If the question is not directly related to routing, you can still help but remind them of your primary purpose.
            - Keep responses concise and focused on helping the user navigate Munich.
            - If you don't know the answer, say so directly and ask if they'd like routing assistance instead.
            
            When thinking through complex problems, explain your reasoning step by step.
            
            If users share images, analyze them for:
            1. Munich landmarks or points of interest
            2. Street signs or transportation information
            3. Maps or directional information
            
            Respond to images by identifying what's in them and how it might relate to navigation or routing in Munich.
            """

            # Construct messages with conversation history
            messages = [{"role": "system", "content": system_prompt}]
            messages.extend(conversation_history)

            # Prepare user message
            user_message = {"role": "user"}

            # If there's an image, add it to the content
            if image_path:
                logger.debug("Processing image for chat message")
                # For multimodal models, the content should be an array
                content = []

                # Add text if available
                if query and query.strip():
                    content.append({"type": "text", "text": query})
                    logger.debug(f"Added text content: {query}")

                # Process and add image
                image_content = self._prepare_image_for_message(image_path)
                if image_content:
                    content.append(image_content)
                    logger.debug("Added image content to message")
                else:
                    logger.warning("Failed to process image for message")

                if not content:
                    # Fallback if no content could be added
                    fallback_text = (
                        "What can you tell me about this location in Munich?"
                    )
                    content = [{"type": "text", "text": fallback_text}]
                    logger.debug(f"Using fallback text content: {fallback_text}")

                user_message["content"] = content
            else:
                # Simple text message
                user_message["content"] = query
                logger.debug(f"Using simple text message: {query}")

            # Add user message
            messages.append(user_message)
            logger.debug(f"Final message count: {len(messages)}")

            if stream_response:
                logger.debug("Using streaming response for chat")
                # Return the streaming response for the caller to process
                return {
                    "stream": True,
                    "response": self._call_api(messages, temperature=0.7, stream=True),
                    "conversation_id": conversation_id,
                    "messages": messages,
                    "query_type": "chat",
                }

            # Get response
            logger.debug("Making non-streaming API call for chat")
            response = self._call_api(
                messages, temperature=0.7
            )  # Higher temperature for more natural conversation
            response_text = response["choices"][0]["message"]["content"]
            logger.debug(f"Received response length: {len(response_text)} characters")

            # Update conversation history
            logger.debug(f"Updating conversation history for {conversation_id}")
            self.conversations[conversation_id].append(user_message)
            self.conversations[conversation_id].append(
                {"role": "assistant", "content": response_text}
            )

            logger.info(
                f"Chat response generated successfully for conversation: {conversation_id}"
            )
            return response_text

        except Exception as e:
            logger.error(f"Error processing chat message: {str(e)}")
            logger.error(traceback.format_exc())
            raise

    def update_conversation_with_stream_response(
        self, conversation_id: str, query: str, response_text: str, query_type: str
    ) -> None:
        """Update conversation history with streamed response after it completes.

        Args:
            conversation_id: The conversation ID
            query: The user query
            response_text: The full response text
            query_type: The type of query (chat, routing, navigation)
        """
        try:
            logger.debug(
                f"Updating conversation with stream response for {conversation_id} ({query_type})"
            )

            if not conversation_id or conversation_id not in self.conversations:
                logger.warning(
                    f"Cannot update conversation history: invalid conversation ID {conversation_id}"
                )
                return

            if query_type == "chat":
                logger.debug("Updating chat conversation history")
                self.conversations[conversation_id].append(
                    {"role": "user", "content": query}
                )
                self.conversations[conversation_id].append(
                    {"role": "assistant", "content": response_text}
                )
            elif query_type == "routing":
                logger.debug("Updating routing conversation history")
                self.conversations[conversation_id].append(
                    {"role": "user", "content": query}
                )
                self.conversations[conversation_id].append(
                    {"role": "assistant", "content": response_text}
                )
            elif query_type == "navigation":
                logger.debug("Updating navigation conversation history")
                self.conversations[conversation_id].append(
                    {
                        "role": "user",
                        "content": "Generate navigation instructions for route.",
                    }
                )
                self.conversations[conversation_id].append(
                    {"role": "assistant", "content": response_text}
                )
            else:
                logger.warning(f"Unknown query type: {query_type}")

            logger.debug(
                f"Conversation {conversation_id} updated with stream response ({len(response_text)} chars)"
            )

        except Exception as e:
            logger.error(f"Error updating conversation with stream response: {str(e)}")
            logger.error(traceback.format_exc())

    def _is_fuzzy_location_query(self, query: str) -> bool:
        """Determine if a query contains fuzzy location references.

        Args:
            query: The routing query

        Returns:
            True if the query contains fuzzy location references
        """
        # Keywords that indicate potential fuzzy location queries
        fuzzy_indicators = [
            "restaurant",
            "cafe",
            "bar",
            "coffee",
            "food",
            "place to eat",
            "place to drink",
            "pub",
            "hotel",
            "accommodation",
            "place to stay",
            "good",
            "best",
            "nice",
            "great",
            "popular",
            "recommended",
            "close to",
            "near",
            "around",
            "by",
            "in the area",
        ]

        query_lower = query.lower()

        # Check if query contains any fuzzy indicators
        return any(indicator in query_lower for indicator in fuzzy_indicators)

    def _handle_fuzzy_location_query(
        self, query: str, conversation_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Handle a query with fuzzy location references.

        Args:
            query: The routing query
            conversation_id: Optional conversation ID for context

        Returns:
            Dictionary of structured routing parameters
        """
        logger.info("Processing fuzzy location query")

        # 1. First, process the query with standard LLM to get high-level structure
        # This gives us information about origin, rough destination area, and travel mode
        standard_params = self._extract_routing_params_with_llm(query, conversation_id)

        if not standard_params:
            logger.warning("Failed to extract any routing parameters with LLM")
            return {"origin": None, "destination": None, "mode": "driving"}

        # 2. Check if destination contains fuzzy references
        destination = standard_params.get("destination", "")
        if not destination:
            logger.warning("No destination found in query")
            return standard_params

        # 3. Parse the fuzzy location query to extract the category and reference location
        category, reference_location, attributes = parse_fuzzy_location(destination)
        logger.debug(
            f"Parsed fuzzy query: category={category}, location={reference_location}, attributes={attributes}"
        )

        if not category:
            # If we couldn't extract a distinct category, just return standard params
            logger.debug("No distinct category found in destination")
            return standard_params

        # If we don't have a reference location but have a category, the entire destination
        # might be just a category, like "a good restaurant"
        if not reference_location and category:
            # Try to determine a logical reference point (e.g., city center, near the origin)
            reference_location = self._determine_reference_location(
                standard_params, query
            )
            logger.info(f"Determined reference location: {reference_location}")

        # 4. Search for places matching the category near the reference location
        logger.info(f"Searching for {category} near {reference_location}")

        # Determine search radius based on the mode of transport
        mode = standard_params.get("mode", "driving")
        search_radius = self._determine_search_radius(mode)

        # Search for places
        place_search = self.place_search_service
        places = place_search.search_nearby(
            category=category,
            near_location=reference_location or "Munich",
            radius=search_radius,
            limit=5,  # Get top 5 results for better selection
        )

        if not places:
            logger.warning(f"No places found for {category} near {reference_location}")
            return standard_params

        # 5. Select the best place based on attributes
        selected_place = self._select_place_by_attributes(places, attributes)
        logger.info(f"Selected place: {selected_place['name']}")

        # 6. Update the parameters with the specific place
        result = standard_params.copy()
        result["destination"] = selected_place["name"]
        result["destination_coords"] = selected_place["coordinates"]
        result["fuzzy_resolution"] = {
            "original_query": destination,
            "resolved_to": selected_place["name"],
            "category": category,
            "attributes": attributes,
            "reference_location": reference_location,
            "all_places": [
                {"name": p["name"], "category": p.get("category", category)}
                for p in places[:3]
            ],
        }

        return result

    def _determine_reference_location(
        self, routing_params: Dict[str, Any], query: str
    ) -> str:
        """Determine a reasonable reference location when none is explicitly specified.

        Args:
            routing_params: The routing parameters extracted so far
            query: The original query

        Returns:
            A reference location
        """
        # 1. First try to use origin as reference point
        origin = routing_params.get("origin")
        if origin:
            return origin

        # 2. Check for city/neighborhood mentions in the query
        # Define Munich neighborhoods and areas in a more maintainable way
        munich_areas = [
            "marienplatz",
            "city center",
            "downtown",
            "central",
            "centrum",
            "old town",
            "altstadt",
            "schwabing",
            "haidhausen",
            "sendling",
            "neuhausen",
            "bogenhausen",
            "lehel",
            "maxvorstadt",
            "ludwigsvorstadt",
            "giesing",
            "english garden",
            "olympic park",
            "odeonsplatz",
            "karlsplatz",
            "stachus",
            "viktualienmarkt",
            "isarvorstadt",
            "glockenbachviertel",
            "westend",
            "ostbahnhof",
            "hauptbahnhof",
            "main station",
            "central station",
        ]

        query_lower = query.lower()
        for area in munich_areas:
            if area in query_lower:
                return area

        # 3. Default to city center
        return "Munich City Center"

    def _determine_search_radius(self, mode: str) -> int:
        """Determine appropriate search radius based on transportation mode.

        Args:
            mode: Transportation mode (driving, walking, cycling)

        Returns:
            Search radius in meters
        """
        # Use the default search radius from the config
        return DEFAULT_SEARCH_RADIUS.get(mode.lower(), DEFAULT_SEARCH_RADIUS["default"])

    def _select_place_by_attributes(
        self, places: List[Dict[str, Any]], attributes: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Select the best place based on attributes using a sophisticated scoring algorithm.

        Args:
            places: List of places with details
            attributes: Dictionary of attributes and their weights/values

        Returns:
            The selected place
        """
        # If only one place, return it
        if len(places) == 1:
            return places[0]

        # Log the selection process
        attribute_list = attributes.get("general", [])
        logger.debug(
            f"Selecting place from {len(places)} options based on attributes: {attribute_list}"
        )

        # Define score weights for different factors
        weights = {
            "cuisine_match": 50,  # 50% importance to cuisine matching - increased priority
            "proximity": 20,  # 20% importance to proximity - reduced priority
            "category_match": 10,  # 10% importance to category matching
            "attribute_match": 15,  # 15% importance to attribute matching
            "name_quality": 5,  # 5% importance to name features
        }

        # Store scores for each place
        place_scores = []

        # Check for cuisine emphasis in attributes
        cuisine_emphasis = False
        cuisine_type = None

        # Check if authenticity is mentioned
        if "authentic" in attribute_list or "traditional" in attribute_list:
            cuisine_emphasis = True
            # Try to identify cuisine type from query
            if "bavarian" in attribute_list or "german" in attribute_list:
                cuisine_type = "bavarian"
            elif "italian" in attribute_list:
                cuisine_type = "italian"
            elif "chinese" in attribute_list:
                cuisine_type = "chinese"
            elif "mexican" in attribute_list:
                cuisine_type = "mexican"
            # Add more cuisine types as needed

        for i, place in enumerate(places):
            scores = {
                "cuisine_match": 0,
                "proximity": 0,
                "category_match": 0,
                "attribute_match": 0,
                "name_quality": 0,
            }

            # Get place tags and name
            tags = place.get("tags", {})
            name_lower = place["name"].lower()

            # 1. Cuisine match score - NEW!
            cuisine_score = 0
            place_cuisine = tags.get("cuisine", "").lower()

            # If we know the desired cuisine type and this place has cuisine info
            if cuisine_type and place_cuisine:
                # Perfect match for explicitly requested cuisine type
                if cuisine_type == place_cuisine:
                    cuisine_score = 100
                # Partial match (e.g., regional when bavarian was requested)
                elif (
                    cuisine_type == "bavarian"
                    and place_cuisine in ["regional", "german"]
                ) or (
                    cuisine_type == "german"
                    and place_cuisine in ["bavarian", "regional"]
                ):
                    cuisine_score = 80
                # Completely wrong cuisine with authenticity requested should score very low
                elif cuisine_emphasis:
                    cuisine_score = 10  # Heavy penalty for incorrect cuisine
                else:
                    cuisine_score = 30  # Some penalty but not as severe
            # If authenticity matters but cuisine is unknown
            elif cuisine_emphasis:
                # Check name for cuisine clues if tag is missing
                if cuisine_type == "bavarian" and any(
                    term in name_lower
                    for term in [
                        "augustiner",
                        "franziskaner",
                        "hofbräu",
                        "löwenbräu",
                        "paulaner",
                    ]
                ):
                    cuisine_score = 70
                else:
                    cuisine_score = 40  # Unknown cuisine is better than wrong cuisine
            else:
                # Default score when cuisine isn't specified
                cuisine_score = 50

            scores["cuisine_match"] = cuisine_score

            # 2. Proximity score (inversely proportional to distance)
            # Normalize to 0-100 range where 100 is closest
            min_distance = min(p["distance"] for p in places)
            max_distance = max(p["distance"] for p in places)
            if max_distance > min_distance:
                normalized_distance = (place["distance"] - min_distance) / (
                    max_distance - min_distance
                )
                scores["proximity"] = 100 * (1 - normalized_distance)
            else:
                scores["proximity"] = 100  # If all places at same distance

            # 3. Category matching score - simplified since we've already filtered by category
            category_lower = place.get("category", "").lower()
            scores["category_match"] = (
                80  # Base score since category was already matched
            )

            # Bonus for exact category match
            if places[0].get("category", "").lower() == category_lower:
                scores["category_match"] += 20

            # 4. Attribute matching score based on detected attributes
            attribute_score = 0

            # Quality score based on attribute counts
            if attributes["quality"] > 0:
                # Name quality heuristics (in a real system, would use ratings)
                quality_terms = [
                    "gourmet",
                    "fine",
                    "authentic",
                    "traditional",
                    "premium",
                    "quality",
                    "speciality",
                    "special",
                    "award",
                    "chef",
                    "signature",
                    "craft",
                    "house",
                    "famous",
                    "best",
                    "popular",
                ]
                if any(term in name_lower for term in quality_terms):
                    attribute_score += 20

                # Bonus for places with longer, more specific names
                if len(name_lower.split()) >= 3:
                    attribute_score += 10

            # Cost score
            if attributes["cost"] > 0:
                cost_detected = False
                # Check if we're looking for expensive places
                if (
                    "expensive" in attribute_list
                    or "luxury" in attribute_list
                    or "upscale" in attribute_list
                ):
                    expensive_indicators = [
                        "gourmet",
                        "fine dining",
                        "deluxe",
                        "premium",
                        "house",
                    ]
                    if any(ind in name_lower for ind in expensive_indicators):
                        attribute_score += 20
                        cost_detected = True

                # Check if we're looking for cheap places
                elif (
                    "cheap" in attribute_list
                    or "affordable" in attribute_list
                    or "budget" in attribute_list
                ):
                    budget_indicators = [
                        "express",
                        "fast",
                        "quick",
                        "budget",
                        "value",
                        "economic",
                    ]
                    if any(ind in name_lower for ind in budget_indicators):
                        attribute_score += 20
                        cost_detected = True

                # If no specific cost attribute was detected but cost was important
                if not cost_detected:
                    attribute_score += 5  # Minor bonus

            # Authenticity score
            if attributes["authenticity"] > 0:
                authenticity_terms = [
                    "authentic",
                    "traditional",
                    "original",
                    "real",
                    "genuine",
                    "local",
                ]

                # Check for authenticity in name
                if any(term in name_lower for term in authenticity_terms):
                    attribute_score += 25

                # Direct OSM tag authenticity match (cuisine-specific)
                if cuisine_type and place_cuisine and cuisine_type == place_cuisine:
                    attribute_score += 40  # Strong bonus for matching cuisine
                # Penalize mismatched cuisines when authenticity requested
                elif cuisine_type and place_cuisine and cuisine_type != place_cuisine:
                    attribute_score -= 30  # Strong penalty for wrong cuisine

            # Speed/Service score
            if attributes["speed"] > 0:
                speed_terms = ["express", "quick", "fast", "rapid", "speedy", "prompt"]
                if any(term in name_lower for term in speed_terms):
                    attribute_score += 25

            # Atmosphere score
            if attributes["atmosphere"] > 0:
                atmosphere_terms = [
                    "cozy",
                    "quiet",
                    "romantic",
                    "family",
                    "friendly",
                    "casual",
                    "elegant",
                ]
                if any(term in name_lower for term in atmosphere_terms):
                    attribute_score += 20

            # Food quality score
            if attributes["food_quality"] > 0:
                food_terms = [
                    "delicious",
                    "tasty",
                    "fresh",
                    "healthy",
                    "organic",
                    "gourmet",
                    "specialty",
                ]
                if any(term in name_lower for term in food_terms):
                    attribute_score += 20

            # Normalize attribute score to 0-100
            scores["attribute_match"] = min(100, max(0, attribute_score * 2))

            # 5. Name quality - prefer names with good descriptors
            name_quality_score = 50  # Base score
            positive_terms = [
                "best",
                "favorite",
                "original",
                "authentic",
                "special",
                "famous",
                "house",
                "traditional",
                "quality",
                "gourmet",
                "craft",
            ]
            if any(term in name_lower for term in positive_terms):
                name_quality_score += 30

            # Avoid generic names
            generic_terms = ["restaurant", "café", "cafe", "bar", "bistro"]
            if name_lower in generic_terms or len(name_lower.split()) == 1:
                name_quality_score -= 20

            scores["name_quality"] = max(0, min(100, name_quality_score))

            # Calculate weighted total score
            total_score = sum(
                score * (weights[category] / 100) for category, score in scores.items()
            )

            # Log detailed score breakdown
            logger.debug(f"Place: {place['name']}")
            logger.debug(
                f"  Cuisine type: {place_cuisine}, Match score: {scores['cuisine_match']:.1f}"
            )
            logger.debug(
                f"  Distance: {place['distance']:.0f}m - Proximity score: {scores['proximity']:.1f}"
            )
            logger.debug(f"  Category match score: {scores['category_match']:.1f}")
            logger.debug(f"  Attribute match score: {scores['attribute_match']:.1f}")
            logger.debug(f"  Name quality score: {scores['name_quality']:.1f}")
            logger.debug(f"  Total weighted score: {total_score:.1f}")

            place_scores.append((place, total_score))

        # Sort by total score descending
        place_scores.sort(key=lambda x: x[1], reverse=True)

        # Log selection result
        best_place, best_score = place_scores[0]
        logger.info(f"Selected place: {best_place['name']} with score {best_score:.1f}")

        return best_place

    def _extract_routing_params_with_llm(
        self, query: str, conversation_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Extract routing parameters using the LLM.

        Args:
            query: The routing query
            conversation_id: Optional conversation ID for context

        Returns:
            Dictionary of structured routing parameters
        """
        # Create a modified system prompt that handles fuzzy locations
        system_prompt = """
        You are a helpful assistant that converts natural language routing queries into structured parameters.
        Extract the origin, destination, and any other relevant parameters from the query.
        For addresses in Munich, return detailed information.
        
        If the destination includes a category like "restaurant", "cafe", etc., include that in the destination field.
        For example, if the query is "how to get to a good restaurant near Marienplatz", the destination should be
        "a good restaurant near Marienplatz".
        
        IMPORTANT: You MUST return your response ONLY as a valid JSON object with these fields:
        - origin: Origin location (address or coordinates)
        - destination: Destination location (address, coordinates, or category + location)
        - mode: Transportation mode (default: "driving")
        - waypoints: List of intermediate waypoints (optional)
        - avoid: List of features to avoid (optional)
        - departure_time: Departure time (optional)
        - arrival_time: Arrival time (optional)
        
        DO NOT provide explanations, details, or any other text outside of the JSON object.
        DO NOT use markdown formatting. ONLY return a valid JSON object.
        
        All addresses should be formatted properly for Munich, Germany.
        """

        # Construct messages
        messages = [{"role": "system", "content": system_prompt}]

        # Add conversation history if available
        if conversation_id:
            conversation_id = self.create_conversation(conversation_id)
            conversation_history = self.conversations.get(conversation_id, [])
            messages.extend(conversation_history)

        # Add the user query
        messages.append({"role": "user", "content": query})

        # Call LLM API
        response = self._call_api(messages)
        response_text = response["choices"][0]["message"]["content"]

        # Extract JSON from response
        try:
            result = None

            # Try direct JSON parsing
            try:
                json_str = response_text
                if "```json" in response_text:
                    json_str = response_text.split("```json")[1].split("```")[0].strip()
                elif "```" in response_text:
                    json_str = response_text.split("```")[1].split("```")[0].strip()

                result = json.loads(json_str)
            except (json.JSONDecodeError, IndexError):
                # Fallback to text extraction
                result = self._extract_routing_info_from_text(response_text)

            return result or {}

        except Exception as e:
            logger.error(f"Error extracting routing parameters: {e}")
            return {}
