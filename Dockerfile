FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy project files
COPY pyproject.toml README.md ./
COPY mcp_routing/ ./mcp_routing/

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir .

# Create necessary directories
RUN mkdir -p data/osrm data/ors

# Expose the service port
EXPOSE 8000

# Set environment variables
ENV PYTHONUNBUFFERED=1

# Run the service
CMD ["python", "-m", "mcp_routing", "server"] 