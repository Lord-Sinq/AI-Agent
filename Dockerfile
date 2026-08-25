# Use Python 3.11 slim image for smaller size
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Set Python environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies (if needed for any packages)
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (for better caching)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy all application code
COPY main.py llms.py agent.py ./

# Create data directory
RUN mkdir -p /app/data

# Create the user with a proper home directory AND bash shell
RUN addgroup --system appgroup && \
    adduser --system --ingroup appgroup --shell /bin/bash --home /home/appuser appuser

# Create and set permissions for VS Code server directory
RUN mkdir -p /home/appuser/.vscode-server && \
    chown -R appuser:appgroup /home/appuser

# Switch to non-root user
USER appuser

# Default command (interactive mode)
CMD ["python", "main.py"]