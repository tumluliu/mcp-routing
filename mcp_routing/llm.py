"""LLM interface for DeepSeek."""

import json
import requests
import base64
from typing import Dict, List, Optional, Any, Union
import os
import traceback
from loguru import logger

from .config import DEEPSEEK_ENDPOINT, DEEPSEEK_API_KEY, DEEPSEEK_MODEL


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
        try:
            logger.info(f"Parsing routing query: {query}")
            if image_path:
                logger.info(f"Query includes image: {image_path}")

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
            
            Return a JSON object with these fields:
            - origin: Origin location (address or coordinates)
            - destination: Destination location (address or coordinates)
            - mode: Transportation mode (default: "driving")
            - waypoints: List of intermediate waypoints (optional)
            - avoid: List of features to avoid (optional)
            - departure_time: Departure time (optional)
            - arrival_time: Arrival time (optional)
            - landmarks: Information about landmarks identified in the image (optional)
            
            All addresses should be formatted properly for Munich, Germany.
            
            If the user references locations from previous conversations (like "there", "that place", 
            "the same location", etc.), use the context from previous messages to determine the actual locations.
            
            If an image is included but no text query is provided, extract as much information as possible from the
            image and respond with information about the landmark and how it can be used for routing.
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
                logger.info(f"Successfully parsed routing query: {list(result.keys())}")
                return result
            except (json.JSONDecodeError, IndexError) as e:
                logger.error(f"Failed to parse LLM response: {str(e)}")
                logger.error(f"Response text: {response_text}")
                raise Exception(
                    f"Failed to parse LLM response: {str(e)}\nResponse: {response_text}"
                )

        except Exception as e:
            logger.error(f"Error parsing routing query: {str(e)}")
            logger.error(traceback.format_exc())
            raise

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
