# Use Python 3.9 slim base image for optimal size and compatibility
FROM python:3.9-slim

# Set metadata
LABEL maintainer="ASMERA NDA Team"
LABEL description="Machine Learning Application with SHAP/LIME Interpretation"
LABEL version="1.0.0"

# Set working directory
WORKDIR /app

# Set environment variables for non-interactive installation
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Install system dependencies required for ML libraries
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    libgomp1 \
    libopenblas-dev \
    liblapack-dev \
    gfortran \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Upgrade pip and install wheel for better package management
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Copy requirements first for better Docker layer caching
COPY docker_requirements.txt requirements.txt

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create necessary directories with proper permissions
RUN mkdir -p /app/models /app/uploads /app/interpretation_cache \
    && chmod 755 /app/models /app/uploads /app/interpretation_cache

# Create non-root user for security
RUN useradd --create-home --shell /bin/bash appuser \
    && chown -R appuser:appuser /app
USER appuser

# Expose Streamlit port
EXPOSE 8501

# Set Streamlit environment variables
ENV STREAMLIT_SERVER_PORT=8501
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0
ENV STREAMLIT_SERVER_HEADLESS=true
ENV STREAMLIT_BROWSER_GATHER_USAGE_STATS=false
ENV PYTHONPATH=/app

# Health check to verify the application is running
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

# Run the application
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
