"""LLM interface for DeepSeek."""

import json
import requests
from typing import Dict, List, Optional, Any, Union

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

    def _call_api(
        self, messages: List[Dict[str, str]], temperature: float = 0.0
    ) -> Dict[str, Any]:
        """Make a raw API call to DeepSeek.

        Args:
            messages: List of message dictionaries
            temperature: Sampling temperature

        Returns:
            API response
        """
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
        }

        response = requests.post(
            f"{self.endpoint}/chat/completions", headers=self.headers, json=payload
        )

        if response.status_code != 200:
            raise Exception(
                f"DeepSeek API error: {response.status_code} - {response.text}"
            )

        return response.json()

    def parse_routing_query(self, query: str) -> Dict[str, Any]:
        """Parse a natural language routing query into structured parameters.

        Args:
            query: The natural language query

        Returns:
            Dictionary of structured routing parameters
        """
        system_prompt = """
        You are a helpful assistant that converts natural language routing queries into structured parameters.
        Extract the origin, destination, and any other relevant parameters from the query.
        For addresses in Munich, return detailed information.
        
        Return a JSON object with these fields:
        - origin: Origin location (address or coordinates)
        - destination: Destination location (address or coordinates)
        - mode: Transportation mode (default: "driving")
        - waypoints: List of intermediate waypoints (optional)
        - avoid: List of features to avoid (optional)
        - departure_time: Departure time (optional)
        - arrival_time: Arrival time (optional)
        
        All addresses should be formatted properly for Munich, Germany.
        """

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query},
        ]

        response = self._call_api(messages)
        response_text = response["choices"][0]["message"]["content"]

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

    def generate_navigation_instructions(self, route_data: Dict[str, Any]) -> List[str]:
        """Generate human-friendly navigation instructions from route data.

        Args:
            route_data: The routing data from the routing engine

        Returns:
            List of navigation instructions
        """
        system_prompt = """
        You are a helpful navigation assistant. Convert the technical routing data into clear, 
        step-by-step navigation instructions that are easy for humans to follow while driving.
        
        Return a list of instructions, one for each major navigation step.
        Focus on important turns, street names, and landmarks.
        Use cardinal directions (North, South, East, West) when appropriate.
        Distances should be in kilometers or meters as appropriate.
        """

        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": f"Generate navigation instructions for this route data:\n{json.dumps(route_data)}",
            },
        ]

        response = self._call_api(messages)
        response_text = response["choices"][0]["message"]["content"]

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
