"""LLM interface for DeepSeek."""

import json
import requests
import base64
from typing import Dict, List, Optional, Any, Union
import os

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
        # Initialize conversation contexts storage
        self.conversations = {}

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
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "stream": stream,
        }

        if stream:
            response = requests.post(
                f"{self.endpoint}/chat/completions",
                headers=self.headers,
                json=payload,
                stream=True,
            )

            if response.status_code != 200:
                raise Exception(
                    f"DeepSeek API error: {response.status_code} - {response.text}"
                )

            return response
        else:
            response = requests.post(
                f"{self.endpoint}/chat/completions", headers=self.headers, json=payload
            )

            if response.status_code != 200:
                raise Exception(
                    f"DeepSeek API error: {response.status_code} - {response.text}"
                )

            return response.json()

    def _prepare_image_for_message(self, image_path: str) -> Dict[str, Any]:
        """Prepare an image to be included in a message.

        Args:
            image_path: Path to the image file

        Returns:
            Image content object to be included in a message
        """
        if not image_path or not os.path.exists(image_path):
            return None

        try:
            # Read image file as base64
            with open(image_path, "rb") as image_file:
                base64_image = base64.b64encode(image_file.read()).decode("utf-8")

            # Create image content object
            # Note: The exact format might vary based on DeepSeek's multimodal API specifics
            # This is based on the OpenAI format
            return {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
            }
        except Exception as e:
            print(f"Error processing image: {e}")
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

        # Initialize conversation with empty history if it doesn't exist
        if conversation_id not in self.conversations:
            self.conversations[conversation_id] = []

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
        # Ensure conversation exists
        if conversation_id:
            conversation_id = self.create_conversation(conversation_id)
            conversation_history = self.conversations.get(conversation_id, [])
        else:
            conversation_history = []

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
            # For multimodal models, the content should be an array
            content = []

            # Add text if available
            if query and query.strip():
                content.append({"type": "text", "text": query})

            # Process and add image
            image_content = self._prepare_image_for_message(image_path)
            if image_content:
                content.append(image_content)

            if not content:
                # Fallback if no content could be added
                content = [
                    {
                        "type": "text",
                        "text": "Please analyze this image for landmarks in Munich.",
                    }
                ]

            user_message["content"] = content
        else:
            # Simple text message
            user_message["content"] = query

        # Add user message
        messages.append(user_message)

        if stream_response:
            # Return the streaming response for the caller to process
            return {
                "stream": True,
                "response": self._call_api(messages, stream=True),
                "conversation_id": conversation_id,
                "messages": messages,
                "query_type": "routing",
            }

        response = self._call_api(messages)
        response_text = response["choices"][0]["message"]["content"]

        # Update conversation history if using a conversation context
        if conversation_id:
            self.conversations[conversation_id].append(user_message)
            self.conversations[conversation_id].append(
                {"role": "assistant", "content": response_text}
            )

        try:
            # Extract JSON from response
            json_str = response_text
            if "```json" in response_text:
                json_str = response_text.split("```json")[1].split("```")[0].strip()
            elif "```" in response_text:
                json_str = response_text.split("```")[1].split("```")[0].strip()

            result = json.loads(json_str)
            return result
        except (json.JSONDecodeError, IndexError) as e:
            raise Exception(
                f"Failed to parse LLM response: {str(e)}\nResponse: {response_text}"
            )

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
        # Ensure conversation exists
        if conversation_id:
            conversation_id = self.create_conversation(conversation_id)
            conversation_history = self.conversations.get(conversation_id, [])
        else:
            conversation_history = []

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

        # Add only relevant conversation history about navigation preferences, if any
        if conversation_history:
            # Filter to only include the most recent messages that might be relevant to navigation
            relevant_messages = conversation_history[
                -4:
            ]  # Last 4 messages (2 exchanges)
            messages.extend(relevant_messages)

        # Add the current request
        messages.append(
            {
                "role": "user",
                "content": f"Generate navigation instructions for this route data:\n{json.dumps(route_data)}",
            }
        )

        if stream_response:
            # Return the streaming response for the caller to process
            return {
                "stream": True,
                "response": self._call_api(messages, stream=True),
                "conversation_id": conversation_id,
                "messages": messages,
                "query_type": "navigation",
            }

        response = self._call_api(messages)
        response_text = response["choices"][0]["message"]["content"]

        # Update conversation history if using a conversation context
        if conversation_id:
            self.conversations[conversation_id].append(
                {
                    "role": "user",
                    "content": f"Generate navigation instructions for route.",
                }
            )
            self.conversations[conversation_id].append(
                {"role": "assistant", "content": response_text}
            )

        # Split by newlines and clean up
        instructions = [
            line.strip() for line in response_text.split("\n") if line.strip()
        ]

        # Remove numbered prefixes if present
        clean_instructions = []
        for instr in instructions:
            if instr[0].isdigit() and instr[1:3] in [". ", ") ", "- "]:
                instr = instr[3:].strip()
            clean_instructions.append(instr)

        return clean_instructions

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
        # Ensure conversation exists
        conversation_id = self.create_conversation(conversation_id)
        conversation_history = self.conversations.get(conversation_id, [])

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
            # For multimodal models, the content should be an array
            content = []

            # Add text if available
            if query and query.strip():
                content.append({"type": "text", "text": query})

            # Process and add image
            image_content = self._prepare_image_for_message(image_path)
            if image_content:
                content.append(image_content)

            if not content:
                # Fallback if no content could be added
                content = [
                    {
                        "type": "text",
                        "text": "What can you tell me about this location in Munich?",
                    }
                ]

            user_message["content"] = content
        else:
            # Simple text message
            user_message["content"] = query

        # Add user message
        messages.append(user_message)

        if stream_response:
            # Return the streaming response for the caller to process
            return {
                "stream": True,
                "response": self._call_api(messages, temperature=0.7, stream=True),
                "conversation_id": conversation_id,
                "messages": messages,
                "query_type": "chat",
            }

        # Get response
        response = self._call_api(
            messages, temperature=0.7
        )  # Higher temperature for more natural conversation
        response_text = response["choices"][0]["message"]["content"]

        # Update conversation history
        self.conversations[conversation_id].append(user_message)
        self.conversations[conversation_id].append(
            {"role": "assistant", "content": response_text}
        )

        return response_text

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
        if not conversation_id or conversation_id not in self.conversations:
            return

        if query_type == "chat":
            self.conversations[conversation_id].append(
                {"role": "user", "content": query}
            )
            self.conversations[conversation_id].append(
                {"role": "assistant", "content": response_text}
            )
        elif query_type == "routing":
            self.conversations[conversation_id].append(
                {"role": "user", "content": query}
            )
            self.conversations[conversation_id].append(
                {"role": "assistant", "content": response_text}
            )
        elif query_type == "navigation":
            self.conversations[conversation_id].append(
                {
                    "role": "user",
                    "content": "Generate navigation instructions for route.",
                }
            )
            self.conversations[conversation_id].append(
                {"role": "assistant", "content": response_text}
            )
