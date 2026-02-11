FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY backend/requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy backend code
COPY backend ./backend

# Copy root config files needed for install (if any)
COPY backend/setup.py .
COPY backend/pyproject.toml .

# Install the package in editable mode or just rely on pythonpath
RUN pip install -e .

# Expose port
EXPOSE 8000

# Create a non-root user and set permissions for logs
RUN useradd -m myuser && \
    mkdir -p /app/backend/logs && \
    chown -R myuser:myuser /app

USER myuser

# Command to run using uvicorn
CMD ["uvicorn", "backend.app.main:app", "--host", "0.0.0.0", "--port", "8000"]
