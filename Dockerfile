# Use a smaller base image
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install CPU-only PyTorch first to prevent GPU version from being installed
RUN pip install --no-cache-dir torch torchvision --extra-index-url https://download.pytorch.org/whl/cpu

# Copy requirements file
COPY requirements.txt .

# Install other dependencies with no cache
RUN pip install --no-cache-dir -r requirements.txt \
    && rm -rf /root/.cache/pip  # Remove pip cache if any

# Copy application files
COPY server.py .
COPY mnist_model.pth .
COPY static ./static/

# Expose the necessary port
EXPOSE 8000

# Run the application
CMD ["python", "server.py"]
