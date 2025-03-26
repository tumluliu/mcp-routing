"""MCP Routing API using FastAPI."""

from fastapi import FastAPI, HTTPException, Request, File, UploadFile, Form
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import os
import uuid
import json
import asyncio
import shutil
import traceback
from pathlib import Path
from loguru import logger
import marked

# Configure loguru
logger.remove()  # Remove default handler
logger.add(
    "logs/mcp_routing.log",
    rotation="10 MB",
    retention="1 week",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {name}:{function}:{line} - {message}",
    level="DEBUG",
    backtrace=True,
    diagnose=True,
)
logger.add(lambda msg: print(msg), level="INFO", format="{level} | {message}")

# Create logs directory if it doesn't exist
logs_dir = Path("logs")
logs_dir.mkdir(exist_ok=True)

from .llm import DeepSeekLLM
from .routing import get_routing_engine
from .visualization import create_route_map
from .config import SERVICE_HOST, SERVICE_PORT, RELOAD

# Create uploads directory if it doesn't exist
UPLOADS_DIR = Path("uploads")
UPLOADS_DIR.mkdir(exist_ok=True)
logger.info(f"Ensuring uploads directory exists at {UPLOADS_DIR}")

app = FastAPI(
    title="MCP Routing Service",
    description="Natural language routing service using DeepSeek and OSM-based routing engines",
    version="0.1.0",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files directory for uploaded images
app.mount("/uploads", StaticFiles(directory="uploads"), name="uploads")
logger.info("Static files directory mounted at /uploads")

# Initialize services
logger.info("Initializing LLM service...")
llm = DeepSeekLLM()
try:
    logger.info("Initializing routing engine...")
    routing_engine = get_routing_engine()
    logger.info(f"Routing engine initialized: {type(routing_engine).__name__}")
except ValueError as e:
    logger.error(f"Routing engine initialization error: {e}")
    # Fallback to prevent server from failing to start
    from .routing import DummyRoutingEngine

    routing_engine = DummyRoutingEngine()
    logger.warning("Fallback to DummyRoutingEngine due to initialization error")


# Add middleware to log request information
@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log incoming requests and their processing time."""
    start_time = asyncio.get_event_loop().time()

    # Generate a request ID
    request_id = str(uuid.uuid4())

    # Log the request
    logger.info(f"Request {request_id} - {request.method} {request.url.path}")

    try:
        # Process the request
        response = await call_next(request)

        # Calculate processing time
        process_time = asyncio.get_event_loop().time() - start_time

        # Log the response
        logger.info(
            f"Request {request_id} completed - Status: {response.status_code} - Time: {process_time:.3f}s"
        )

        return response
    except Exception as e:
        # Log the exception
        logger.error(f"Request {request_id} failed: {str(e)}")
        logger.error(f"Exception details: {traceback.format_exc()}")

        # Re-raise the exception to let FastAPI handle it
        raise


class RoutingQuery(BaseModel):
    """Model for routing query requests."""

    query: str
    conversation_id: Optional[str] = None
    stream: bool = False
    image_path: Optional[str] = None


class ChatQuery(BaseModel):
    """Model for chat requests."""

    message: str
    conversation_id: Optional[str] = None
    stream: bool = False
    image_path: Optional[str] = None


class RoutingResult(BaseModel):
    """Model for routing query results."""

    query: str
    parsed_params: Dict[str, Any]
    route_data: Dict[str, Any]
    instructions: List[str]
    map_url: str
    conversation_id: str
    fuzzy_resolution: Optional[Dict[str, Any]] = None


class ChatResult(BaseModel):
    """Model for chat responses."""

    message: str
    response: str
    conversation_id: str
    is_routing_query: bool = False


class ImageUploadResult(BaseModel):
    """Model for image upload results."""

    image_path: str
    image_url: str
    success: bool
    message: str


async def stream_processor(response, query_type):
    """Process a streaming response into an SSE compatible format.

    Args:
        response: Streaming response from DeepSeek
        query_type: Type of query being processed

    Yields:
        Server-sent event formatted strings
    """
    full_text = ""
    last_send_time = 0

    for line in response.iter_lines():
        if not line:
            continue

        try:
            line_text = line.decode("utf-8")
            if line_text.startswith("data: "):
                data = line_text[6:]  # Remove 'data: ' prefix
                if data == "[DONE]":
                    break

                try:
                    chunk = json.loads(data)
                    if chunk and "choices" in chunk and len(chunk["choices"]) > 0:
                        delta = chunk["choices"][0].get("delta", {})
                        content = delta.get("content", "")
                        if content:
                            full_text += content
                            # Send a progress update
                            yield f"data: {json.dumps({'type': 'thinking', 'content': content})}\n\n"
                except json.JSONDecodeError:
                    logger.warning(
                        f"Failed to decode JSON in streaming response: {data}"
                    )
                    continue
        except Exception as e:
            logger.error(f"Error processing streaming response: {str(e)}")
            logger.error(traceback.format_exc())
            yield f"data: {json.dumps({'type': 'error', 'content': str(e)})}\n\n"
            break

    # Send complete message
    yield f"data: {json.dumps({'type': 'complete', 'content': full_text})}\n\n"
    logger.debug(
        f"Stream processing complete for {query_type} query. Full response length: {len(full_text)}"
    )


@app.get("/", response_class=HTMLResponse)
async def root():
    """Root endpoint that provides a simple UI."""
    return """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Munich Route Planner</title>
        <link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;500;700&display=swap">
        <script src="https://cdn.jsdelivr.net/npm/marked/marked.min.js"></script>
        <style>
            :root {
                --primary-color: #0069c0;
                --primary-dark: #004c8c;
                --primary-light: #5694f1;
                --accent-color: #ffc107;
                --text-primary: #212121;
                --text-secondary: #757575;
                --gray-light: #f5f5f5;
                --gray-medium: #e0e0e0;
                --white: #ffffff;
                --shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
                --radius: 8px;
                --transition: all 0.3s ease;
            }
            
            * {
                margin: 0;
                padding: 0;
                box-sizing: border-box;
            }
            
            body {
                font-family: 'Roboto', -apple-system, BlinkMacSystemFont, sans-serif;
                line-height: 1.6;
                color: var(--text-primary);
                background-color: var(--gray-light);
                max-width: 1200px;
                margin: 0 auto;
                padding: 20px;
            }
            
            header {
                margin-bottom: 30px;
                padding-bottom: 20px;
                border-bottom: 1px solid var(--gray-medium);
            }
            
            h1 {
                color: var(--primary-color);
                font-size: 32px;
                font-weight: 500;
                margin-bottom: 10px;
            }
            
            p {
                color: var(--text-secondary);
                margin-bottom: 15px;
            }
            
            .main-container {
                display: grid;
                grid-template-columns: 1fr;
                gap: 20px;
            }
            
            @media (min-width: 768px) {
                .main-container {
                    grid-template-columns: 1fr 1fr;
                }
            }
            
            .chat-section, .map-section {
                background-color: var(--white);
                border-radius: var(--radius);
                box-shadow: var(--shadow);
                padding: 20px;
                min-height: 500px;
            }
            
            .chat-section {
                display: flex;
                flex-direction: column;
            }
            
            .controls {
                margin-bottom: 15px;
            }
            
            .toggle-container {
                display: flex;
                align-items: center;
                margin-bottom: 15px;
            }
            
            .toggle-switch {
                position: relative;
                display: inline-block;
                width: 50px;
                height: 24px;
                margin-left: 10px;
            }
            
            .toggle-switch input { 
                opacity: 0;
                width: 0;
                height: 0;
            }
            
            .toggle-slider {
                position: absolute;
                cursor: pointer;
                top: 0;
                left: 0;
                right: 0;
                bottom: 0;
                background-color: var(--gray-medium);
                transition: var(--transition);
                border-radius: 24px;
            }
            
            .toggle-slider:before {
                position: absolute;
                content: "";
                height: 18px;
                width: 18px;
                left: 3px;
                bottom: 3px;
                background-color: var(--white);
                transition: var(--transition);
                border-radius: 50%;
            }
            
            input:checked + .toggle-slider {
                background-color: var(--primary-color);
            }
            
            input:checked + .toggle-slider:before {
                transform: translateX(26px);
            }
            
            .toggle-label {
                font-size: 14px;
                color: var(--text-secondary);
            }
            
            #chat-container { 
                flex: 1;
                overflow-y: auto;
                padding: 10px;
                margin-bottom: 15px;
                border: 1px solid var(--gray-medium);
                border-radius: var(--radius);
                background-color: var(--gray-light);
                display: flex;
                flex-direction: column;
                min-height: 350px;
                max-height: 500px;
            }
            
            .message { 
                margin: 5px 0;
                padding: 12px 16px;
                border-radius: var(--radius);
                max-width: 80%;
                animation: fadeIn 0.3s ease;
            }
            
            .message-content {
                line-height: 1.5;
            }
            
            .message-content p {
                margin: 0.5em 0;
            }
            
            .message-content ul, .message-content ol {
                margin: 0.5em 0;
                padding-left: 1.5em;
            }
            
            .message-content li {
                margin: 0.3em 0;
            }
            
            .message-content code {
                background-color: rgba(0, 0, 0, 0.1);
                padding: 0.2em 0.4em;
                border-radius: 3px;
                font-family: monospace;
            }
            
            .message-content pre {
                background-color: rgba(0, 0, 0, 0.1);
                padding: 1em;
                border-radius: var(--radius);
                overflow-x: auto;
                margin: 0.5em 0;
            }
            
            .message-content pre code {
                background-color: transparent;
                padding: 0;
            }
            
            .message-content blockquote {
                border-left: 3px solid var(--primary-color);
                margin: 0.5em 0;
                padding-left: 1em;
                color: var(--text-secondary);
            }
            
            .message-content table {
                border-collapse: collapse;
                width: 100%;
                margin: 0.5em 0;
            }
            
            .message-content th, .message-content td {
                border: 1px solid var(--gray-medium);
                padding: 0.5em;
                text-align: left;
            }
            
            .message-content th {
                background-color: var(--gray-light);
            }
            
            .message-content a {
                color: var(--primary-color);
                text-decoration: none;
            }
            
            .message-content a:hover {
                text-decoration: underline;
            }
            
            @keyframes fadeIn {
                from { opacity: 0; transform: translateY(10px); }
                to { opacity: 1; transform: translateY(0); }
            }
            
            .user-message {
                background-color: var(--primary-light);
                color: var(--white);
                align-self: flex-end;
                border-bottom-right-radius: 2px;
            }
            
            .bot-message {
                background-color: var(--white);
                color: var(--text-primary);
                align-self: flex-start;
                border-bottom-left-radius: 2px;
                box-shadow: var(--shadow);
            }
            
            .thinking {
                font-style: italic;
                color: var(--text-secondary);
                background-color: #f0f0f0;
            }
            
            .typing-indicator {
                display: inline-block;
                width: 20px;
                height: 10px;
                position: relative;
                margin-left: 5px;
            }
            
            .typing-indicator span {
                background-color: var(--text-secondary);
                display: block;
                float: left;
                width: 4px;
                height: 4px;
                border-radius: 50%;
                margin: 3px 1px;
                animation: typing 1s infinite ease-in-out;
            }
            
            .typing-indicator span:nth-child(1) { animation-delay: 0.2s; }
            .typing-indicator span:nth-child(2) { animation-delay: 0.4s; }
            .typing-indicator span:nth-child(3) { animation-delay: 0.6s; }
            
            @keyframes typing {
                0% { transform: translateY(0px); }
                50% { transform: translateY(-5px); }
                100% { transform: translateY(0px); }
            }
            
            .input-container {
                display: flex;
                margin-top: 10px;
                align-items: center;
                position: relative;
            }
            
            .input-wrapper {
                flex: 1;
                position: relative;
            }
            
            #query {
                width: 100%;
                padding: 12px 12px 12px 45px;
                border: 1px solid var(--gray-medium);
                border-radius: 24px;
                font-size: 15px;
                outline: none;
                transition: var(--transition);
            }
            
            #query:focus {
                border-color: var(--primary-color);
                box-shadow: 0 0 0 2px rgba(0, 105, 192, 0.2);
            }
            
            .image-upload-container {
                position: absolute;
                left: 12px;
                top: 50%;
                transform: translateY(-50%);
            }
            
            .image-upload-button {
                background: none;
                border: none;
                cursor: pointer;
                font-size: 18px;
                color: var(--text-secondary);
                display: flex;
                align-items: center;
                justify-content: center;
                width: 24px;
                height: 24px;
                padding: 0;
                transition: var(--transition);
            }
            
            .image-upload-button:hover {
                color: var(--primary-color);
            }
            
            #file-input {
                display: none;
            }
            
            .send-button {
                background-color: var(--primary-color);
                color: white;
                border: none;
                border-radius: 24px;
                padding: 10px 20px;
                margin-left: 8px;
                cursor: pointer;
                font-weight: 500;
                transition: var(--transition);
            }
            
            .send-button:hover {
                background-color: var(--primary-dark);
            }
            
            .image-preview {
                max-width: 200px;
                max-height: 150px;
                border-radius: var(--radius);
                margin: 5px 0;
                border: 1px solid var(--gray-medium);
            }
            
            .image-in-message {
                max-width: 250px;
                max-height: 200px;
                border-radius: 4px;
                margin: 5px 0;
            }
            
            .message-with-image {
                display: flex;
                flex-direction: column;
            }
            
            .image-caption {
                margin-top: 5px;
                font-style: italic;
                font-size: 0.9em;
                color: var(--text-secondary);
            }
            
            .image-status {
                font-size: 12px;
                color: var(--text-secondary);
                position: absolute;
                right: 12px;
                top: 50%;
                transform: translateY(-50%);
                transition: var(--transition);
            }
            
            #image-preview-container {
                margin-top: 10px;
            }
            
            #result {
                margin-top: 20px;
                background-color: var(--white);
                border-radius: var(--radius);
                padding: 15px;
                box-shadow: var(--shadow);
            }
            
            #map {
                height: 500px;
                width: 100%;
                border-radius: var(--radius);
                overflow: hidden;
                box-shadow: var(--shadow);
                margin-top: 20px;
            }
            
            #instructions {
                margin-top: 20px;
                background-color: var(--white);
                border-radius: var(--radius);
                padding: 15px;
                box-shadow: var(--shadow);
            }
            
            #instructions h3 {
                color: var(--primary-color);
                margin-bottom: 10px;
            }
            
            #instructions ol {
                padding-left: 20px;
            }
            
            #instructions li {
                margin-bottom: 8px;
            }
            
            .result-card {
                background-color: var(--white);
                border-radius: var(--radius);
                box-shadow: var(--shadow);
                padding: 15px;
                margin-bottom: 15px;
            }
            
            .result-card h3 {
                color: var(--primary-color);
                margin-bottom: 10px;
            }
            
            .card-info {
                display: flex;
                gap: 20px;
                margin-bottom: 15px;
            }
            
            .info-item {
                flex: 1;
            }
            
            .info-label {
                font-size: 12px;
                color: var(--text-secondary);
                margin-bottom: 5px;
            }
            
            .info-value {
                font-size: 18px;
                font-weight: 500;
            }
            
            .info-value.distance {
                color: var(--primary-color);
            }
            
            .info-value.duration {
                color: var(--accent-color);
            }
            
            .map-link {
                display: inline-block;
                margin-top: 10px;
                color: var(--primary-color);
                text-decoration: none;
                font-weight: 500;
                transition: var(--transition);
            }
            
            .map-link:hover {
                color: var(--primary-dark);
                text-decoration: underline;
            }
            
            .fuzzy-resolution {
                margin-top: 15px;
                padding-top: 10px;
                border-top: 1px dashed var(--gray-medium);
            }
            
            .fuzzy-detail {
                font-size: 14px;
                margin: 5px 0;
            }
            
            .highlight {
                color: var(--primary-color);
                font-weight: 500;
            }
            
            /* Responsive adjustments */
            @media (max-width: 768px) {
                .main-container {
                    grid-template-columns: 1fr;
                }
                
                .chat-section, .map-section {
                    min-height: auto;
                }
                
                #chat-container {
                    max-height: 350px;
                }
            }
        </style>
    </head>
    <body>
        <header>
            <h1>Munich Route Planner</h1>
            <p>Ask for directions, upload landmark images, or chat about navigation in Munich.</p>
        </header>
        
        <div class="main-container">
            <section class="chat-section">
                <div class="controls">
                    <div class="toggle-container">
                        <span class="toggle-label">Show thinking progress:</span>
                        <label class="toggle-switch">
                            <input type="checkbox" id="show-thinking" checked>
                            <span class="toggle-slider"></span>
                        </label>
                    </div>
                </div>
                
                <div id="chat-container"></div>
                
                <div class="input-container">
                    <div class="input-wrapper">
                        <div class="image-upload-container">
                            <label for="file-input" class="image-upload-button">
                                📎
                            </label>
                            <input type="file" id="file-input" accept="image/*" onchange="handleImageUpload(event)">
                        </div>
                        <input type="text" id="query" placeholder="Type your question or routing request..." onkeydown="if(event.key==='Enter') submitQuery()">
                        <span id="image-status" class="image-status"></span>
                    </div>
                    <button class="send-button" onclick="submitQuery()">Send</button>
                </div>
                
                <div id="image-preview-container"></div>
            </section>
            
            <section class="map-section">
                <div id="result"></div>
                <div id="map"></div>
                <div id="instructions"></div>
            </section>
        </div>
        
        <script>
            // Store conversation ID and current image
            let conversationId = null;
            let currentStreamingMessageId = null;
            let currentImagePath = null;
            
            function addMessage(text, isUser, isThinking = false, imagePath = null) {
                const chatContainer = document.getElementById('chat-container');
                
                // If this is a thinking update and we have an existing message to update
                if (isThinking && currentStreamingMessageId) {
                    const existingMessage = document.getElementById(currentStreamingMessageId);
                    if (existingMessage) {
                        existingMessage.innerHTML = text;
                        // Auto-scroll to the bottom
                        chatContainer.scrollTop = chatContainer.scrollHeight;
                        return currentStreamingMessageId;
                    }
                }
                
                // Create new message
                const messageId = 'msg-' + Date.now();
                const messageDiv = document.createElement('div');
                messageDiv.className = isUser ? 'message user-message' : 'message bot-message';
                if (isThinking) {
                    messageDiv.className += ' thinking';
                }
                messageDiv.id = messageId;
                
                // Add image if provided
                if (imagePath) {
                    messageDiv.className += ' message-with-image';
                    const imgElement = document.createElement('img');
                    imgElement.src = imagePath;
                    imgElement.className = 'image-in-message';
                    imgElement.alt = 'Uploaded image';
                    messageDiv.appendChild(imgElement);
                    
                    // Add caption if there's text
                    if (text && text.trim()) {
                        const textElement = document.createElement('div');
                        textElement.className = 'message-content';
                        textElement.innerHTML = marked.parse(text);
                        messageDiv.appendChild(textElement);
                    }
                } else {
                    // Create content div for markdown rendering
                    const contentDiv = document.createElement('div');
                    contentDiv.className = 'message-content';
                    contentDiv.innerHTML = marked.parse(text);
                    messageDiv.appendChild(contentDiv);
                }
                
                chatContainer.appendChild(messageDiv);
                
                // Auto-scroll to the bottom
                chatContainer.scrollTop = chatContainer.scrollHeight;
                return messageId;
            }
            
            function createTypingIndicator() {
                return '<div class="typing-indicator"><span></span><span></span><span></span></div>';
            }
            
            // Add welcome message
            window.onload = function() {
                addMessage('Hello! I can help you find routes in Munich or answer questions about navigation. You can also upload images of landmarks to help me identify locations.', false);
            }
            
            // Handle image upload
            async function handleImageUpload(event) {
                const file = event.target.files[0];
                if (!file) return;
                
                // Show loading state
                const statusElement = document.getElementById('image-status');
                statusElement.textContent = 'Uploading...';
                
                // Create form data
                const formData = new FormData();
                formData.append('file', file);
                
                try {
                    // Upload image
                    const response = await fetch('/upload-image', {
                        method: 'POST',
                        body: formData
                    });
                    
                    const result = await response.json();
                    
                    if (result.success) {
                        // Store the image path
                        currentImagePath = result.image_path;
                        
                        // Show preview
                        const previewContainer = document.getElementById('image-preview-container');
                        previewContainer.innerHTML = '';
                        
                        const previewImg = document.createElement('img');
                        previewImg.src = result.image_url;
                        previewImg.className = 'image-preview';
                        previewContainer.appendChild(previewImg);
                        
                        // Update status
                        statusElement.textContent = 'Image added';
                        statusElement.style.color = '#2e8b57';
                    } else {
                        throw new Error(result.message);
                    }
                } catch (error) {
                    // Show error
                    statusElement.textContent = 'Upload failed: ' + error.message;
                    statusElement.style.color = '#cc3333';
                    currentImagePath = null;
                }
            }
            
            async function submitQuery() {
                const query = document.getElementById('query').value;
                if (!query && !currentImagePath) {
                    alert('Please enter a query or upload an image');
                    return;
                }
                
                // Clear input field
                document.getElementById('query').value = '';
                
                // Get image path if any
                const imagePath = currentImagePath;
                const imageUrl = imagePath ? ('/uploads/' + imagePath.split('/').pop()) : null;
                
                // Add user message to chat with image if applicable
                addMessage(query, true, false, imageUrl);
                
                // Reset image preview and path
                document.getElementById('image-preview-container').innerHTML = '';
                document.getElementById('image-status').textContent = '';
                
                // Check if we should show thinking
                const showThinking = document.getElementById('show-thinking').checked;
                
                if (showThinking) {
                    // Add a preliminary bot message with thinking indicator
                    currentStreamingMessageId = addMessage('Thinking' + createTypingIndicator(), false, true);
                }
                
                try {
                    // First try as a chat message
                    const chatResponse = await fetch('/chat', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ 
                            message: query,
                            conversation_id: conversationId,
                            stream: showThinking,
                            image_path: imagePath
                        })
                    });
                    
                    let responseData;
                    
                    if (showThinking) {
                        // Handle streaming response
                        const reader = chatResponse.body.getReader();
                        const decoder = new TextDecoder();
                        let accumulatedResponse = '';
                        
                        while (true) {
                            const { value, done } = await reader.read();
                            if (done) break;
                            
                            const chunk = decoder.decode(value, { stream: true });
                            const events = chunk.split('\\n\\n');
                            
                            for (const event of events) {
                                if (!event.trim() || !event.startsWith('data: ')) continue;
                                
                                try {
                                    const eventData = JSON.parse(event.substring(6)); // Remove 'data: ' prefix
                                    
                                    if (eventData.type === 'thinking') {
                                        accumulatedResponse += eventData.content;
                                        // Update thinking message
                                        document.getElementById(currentStreamingMessageId).innerHTML = accumulatedResponse;
                                    } else if (eventData.type === 'complete') {
                                        // Final message content
                                        responseData = {
                                            message: query,
                                            response: eventData.content,
                                            conversation_id: conversationId || 'new-conversation',
                                            is_routing_query: false
                                        };
                                        
                                        // Detect if this is a routing query based on keywords
                                        const routingKeywords = [
                                            "route", "path", "directions", "navigate", "get to", "how to get", 
                                            "from", "to", "travel", "drive", "walk", "bus", "subway", "train",
                                            "marienplatz", "english garden", "olympiapark", "central station"
                                        ];
                                        
                                        const lowerCaseResponse = eventData.content.toLowerCase();
                                        if (routingKeywords.some(keyword => query.toLowerCase().includes(keyword))) {
                                            responseData.is_routing_query = true;
                                        }
                                        
                                        // Replace thinking message with final response
                                        document.getElementById(currentStreamingMessageId).innerHTML = eventData.content;
                                        document.getElementById(currentStreamingMessageId).classList.remove('thinking');
                                    }
                                } catch (error) {
                                    console.error('Error parsing event:', error);
                                }
                            }
                        }
                    } else {
                        // Handle regular response
                        responseData = await chatResponse.json();
                        // Add bot response to chat
                        addMessage(responseData.response, false);
                    }
                    
                    // Save conversation ID
                    conversationId = responseData.conversation_id;
                    currentStreamingMessageId = null;
                    
                    // Reset current image path after it's been used
                    currentImagePath = null;
                    
                    // If it's a routing query, also handle routing
                    if (responseData.is_routing_query) {
                        document.getElementById('result').innerHTML = '<div class="result-card"><p>Finding the best route...</p></div>';
                        
                        try {
                            const routeResponse = await fetch('/route', {
                                method: 'POST',
                                headers: { 'Content-Type': 'application/json' },
                                body: JSON.stringify({ 
                                    query: query,
                                    conversation_id: conversationId,
                                    stream: false, // Don't stream the routing execution
                                    image_path: imagePath
                                })
                            });
                            
                            const routeData = await routeResponse.json();
                            
                            // Display route details
                            const distance = (routeData.route_data.distance / 1000).toFixed(1);
                            const duration = Math.round(routeData.route_data.duration / 60);
                            
                            // Construct result HTML
                            let resultHTML = `
                                <div class="result-card">
                                    <h3>Route Found</h3>
                                    <div class="card-info">
                                        <div class="info-item">
                                            <div class="info-label">Distance</div>
                                            <div class="info-value distance">${distance} km</div>
                                        </div>
                                        <div class="info-item">
                                            <div class="info-label">Duration</div>
                                            <div class="info-value duration">${duration} min</div>
                                        </div>
                                    </div>`;
                            
                            // Add fuzzy resolution info if available
                            if (routeData.fuzzy_resolution) {
                                const fuzzy = routeData.fuzzy_resolution;
                                resultHTML += `
                                    <div class="fuzzy-resolution">
                                        <div class="info-label">Destination Resolution</div>
                                        <div class="fuzzy-detail">Found: <span class="highlight">${fuzzy.resolved_to}</span></div>
                                        <div class="fuzzy-detail">Category: ${fuzzy.category}</div>
                                        ${fuzzy.attributes.length > 0 ? 
                                            `<div class="fuzzy-detail">Attributes: ${fuzzy.attributes.join(', ')}</div>` : ''}
                                    </div>`;
                            }
                            
                            resultHTML += `<a href="${routeData.map_url}" target="_blank" class="map-link">Open map in new window</a>
                                </div>`;
                            
                            document.getElementById('result').innerHTML = resultHTML;
                            
                            // Display instructions
                            const instructionsDiv = document.getElementById('instructions');
                            instructionsDiv.innerHTML = '<div class="result-card"><h3>Navigation Instructions</h3><ol>';
                            routeData.instructions.forEach(instruction => {
                                instructionsDiv.innerHTML += `<li>${instruction}</li>`;
                            });
                            instructionsDiv.innerHTML += '</ol></div>';
                            
                            // Load map in iframe
                            document.getElementById('map').innerHTML = `<iframe src="${routeData.map_url}" width="100%" height="500px" frameborder="0"></iframe>`;
                        } catch (error) {
                            addMessage(`I couldn't process that as a routing request. Let me know if you need directions.`, false);
                            document.getElementById('result').innerHTML = '';
                            document.getElementById('map').innerHTML = '';
                            document.getElementById('instructions').innerHTML = '';
                        }
                    } else {
                        // Clear previous routing results if not a routing query
                        document.getElementById('result').innerHTML = '';
                        document.getElementById('map').innerHTML = '';
                        document.getElementById('instructions').innerHTML = '';
                    }
                } catch (error) {
                    if (currentStreamingMessageId) {
                        document.getElementById(currentStreamingMessageId).innerHTML = 'Sorry, I encountered an error while processing your request.';
                        document.getElementById(currentStreamingMessageId).classList.remove('thinking');
                        currentStreamingMessageId = null;
                    } else {
                        addMessage(`Sorry, I encountered an error: ${error.message}`, false);
                    }
                    // Reset current image path after error
                    currentImagePath = null;
                }
            }
        </script>
    </body>
    </html>
    """


@app.post("/upload-image", response_model=ImageUploadResult)
async def upload_image(file: UploadFile = File(...)):
    """Upload an image file.

    Args:
        file: The uploaded file

    Returns:
        ImageUploadResult: Result of the upload operation
    """
    try:
        logger.info(f"Processing image upload: {file.filename}")
        # Generate a unique filename
        filename = f"{uuid.uuid4()}_{file.filename}"
        file_path = UPLOADS_DIR / filename

        # Save the file
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        logger.info(f"Image saved to {file_path}")

        return ImageUploadResult(
            image_path=str(file_path),
            image_url=f"/uploads/{filename}",
            success=True,
            message="Image uploaded successfully",
        )
    except Exception as e:
        logger.error(f"Image upload failed: {str(e)}")
        logger.error(traceback.format_exc())
        return ImageUploadResult(
            image_path="",
            image_url="",
            success=False,
            message=f"Failed to upload image: {str(e)}",
        )


@app.post("/route", response_model=RoutingResult)
async def route(query_data: RoutingQuery):
    """Process a natural language routing query.

    Args:
        query_data: The routing query

    Returns:
        RoutingResult: The routing result
    """
    try:
        logger.info(f"Processing routing query: {query_data.query}")
        # Ensure we have a conversation ID
        conversation_id = query_data.conversation_id or str(uuid.uuid4())
        logger.debug(f"Using conversation ID: {conversation_id}")

        # Check if we need to stream the response
        if query_data.stream:
            logger.debug("Streaming response requested")
            # Get streaming response
            stream_data = llm.parse_routing_query(
                query_data.query,
                conversation_id,
                stream_response=True,
                image_path=query_data.image_path,
            )

            # Return streaming response
            return StreamingResponse(
                stream_processor(stream_data["response"], "routing"),
                media_type="text/event-stream",
            )

        # Parse the natural language query
        logger.debug("Parsing routing query with LLM")
        parsed_params = llm.parse_routing_query(
            query_data.query, conversation_id, image_path=query_data.image_path
        )
        logger.debug(f"Parsed parameters: {parsed_params}")

        # Check for required parameters
        if "origin" not in parsed_params or "destination" not in parsed_params:
            error_msg = "Could not extract origin and destination from query"
            logger.error(error_msg)
            raise HTTPException(
                status_code=400,
                detail=error_msg,
            )

        # Store any fuzzy resolution information
        fuzzy_resolution = parsed_params.pop("fuzzy_resolution", None)
        if fuzzy_resolution:
            logger.info(
                f"Fuzzy location resolved to: {fuzzy_resolution['resolved_to']}"
            )

        # Get routing data
        logger.debug(
            f"Requesting route from {parsed_params['origin']} to {parsed_params['destination']}"
        )

        # Use coordinates directly if available from fuzzy resolution
        if "destination_coords" in parsed_params:
            logger.debug(
                f"Using resolved coordinates for destination: {parsed_params['destination_coords']}"
            )
            destination = parsed_params.pop("destination_coords")
        else:
            destination = parsed_params["destination"]

        route_data = routing_engine.route(
            origin=parsed_params["origin"],
            destination=destination,
            mode=parsed_params.get("mode", "driving"),
            waypoints=parsed_params.get("waypoints"),
            avoid=parsed_params.get("avoid"),
        )
        logger.debug(f"Route data received: {len(route_data.get('steps', []))} steps")

        # Generate navigation instructions (stream or not based on query_data.stream)
        if query_data.stream:
            logger.debug("Streaming navigation instructions")
            # Get streaming response for navigation instructions
            stream_data = llm.generate_navigation_instructions(
                route_data, conversation_id, stream_response=True
            )

            # Return streaming response
            return StreamingResponse(
                stream_processor(stream_data["response"], "navigation"),
                media_type="text/event-stream",
            )
        else:
            # Generate without streaming
            logger.debug("Generating navigation instructions")
            instructions = llm.generate_navigation_instructions(
                route_data, conversation_id
            )
            logger.debug(f"Generated {len(instructions)} instructions")

        # Create map visualization
        logger.debug("Creating route map visualization")
        map_path = create_route_map(route_data, instructions)
        map_url = f"/map/{os.path.basename(map_path)}"
        logger.info(f"Map created: {map_url}")

        # Store the map path temporarily
        app.state.temp_maps = getattr(app.state, "temp_maps", {})
        app.state.temp_maps[os.path.basename(map_path)] = map_path

        result = RoutingResult(
            query=query_data.query,
            parsed_params=parsed_params,
            route_data=route_data,
            instructions=instructions,
            map_url=map_url,
            conversation_id=conversation_id,
            fuzzy_resolution=fuzzy_resolution,
        )
        logger.info(
            f"Route query completed successfully. Conversation ID: {conversation_id}"
        )
        return result

    except HTTPException:
        # Re-raise HTTP exceptions as they are already formatted properly
        raise
    except Exception as e:
        error_msg = f"Error processing routing query: {str(e)}"
        logger.error(error_msg)
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=error_msg)


@app.post("/chat")
async def chat(query_data: ChatQuery):
    """Process a chat message.

    Args:
        query_data: The chat query

    Returns:
        ChatResult or StreamingResponse: The chat response
    """
    try:
        logger.info(f"Processing chat message: {query_data.message}")
        # Ensure we have a conversation ID
        conversation_id = query_data.conversation_id or str(uuid.uuid4())
        logger.debug(f"Using conversation ID: {conversation_id}")

        # Check if we need to stream the response
        if query_data.stream:
            logger.debug("Streaming response requested")
            # Get streaming response
            stream_data = llm.chat(
                query_data.message,
                conversation_id,
                stream_response=True,
                image_path=query_data.image_path,
            )

            # Return streaming response
            return StreamingResponse(
                stream_processor(stream_data["response"], "chat"),
                media_type="text/event-stream",
            )

        # Process the chat message without streaming
        logger.debug("Processing chat message with LLM")
        response = llm.chat(
            query_data.message, conversation_id, image_path=query_data.image_path
        )
        logger.debug(f"Chat response received, length: {len(response)}")

        # Determine if this might be a routing query
        is_routing_query = False

        # Simple heuristic: check if the message contains routing-related keywords
        routing_keywords = [
            "route",
            "path",
            "directions",
            "navigate",
            "get to",
            "how to get",
            "from",
            "to",
            "travel",
            "drive",
            "walk",
            "bus",
            "subway",
            "train",
            "marienplatz",
            "english garden",
            "olympiapark",
            "central station",
        ]

        message_lower = query_data.message.lower()
        if any(keyword in message_lower for keyword in routing_keywords):
            is_routing_query = True
            logger.debug("Message identified as potential routing query")

        result = ChatResult(
            message=query_data.message,
            response=response,
            conversation_id=conversation_id,
            is_routing_query=is_routing_query,
        )
        logger.info(
            f"Chat query completed successfully. Conversation ID: {conversation_id}"
        )
        return result

    except Exception as e:
        error_msg = f"Error processing chat: {str(e)}"
        logger.error(error_msg)
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=error_msg)


@app.get("/map/{map_id}", response_class=HTMLResponse)
async def get_map(map_id: str):
    """Get a generated map.

    Args:
        map_id: The map ID

    Returns:
        HTMLResponse: The HTML map
    """
    try:
        logger.info(f"Retrieving map: {map_id}")
        temp_maps = getattr(app.state, "temp_maps", {})
        map_path = temp_maps.get(map_id)

        if not map_path or not os.path.exists(map_path):
            error_msg = "Map not found"
            logger.warning(f"Map not found: {map_id}")
            raise HTTPException(status_code=404, detail=error_msg)

        with open(map_path, "r") as f:
            map_html = f.read()

        logger.debug(f"Map retrieved successfully: {map_id}")
        return HTMLResponse(content=map_html)
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        error_msg = f"Error retrieving map: {str(e)}"
        logger.error(error_msg)
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=error_msg)


def start_server():
    """Start the FastAPI server."""
    import uvicorn

    logger.info(f"Starting server on {SERVICE_HOST}:{SERVICE_PORT} (reload={RELOAD})")
    uvicorn.run(
        "mcp_routing.api:app", host=SERVICE_HOST, port=SERVICE_PORT, reload=RELOAD
    )
