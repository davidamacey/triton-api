FROM python:3.11

# Set working directory
WORKDIR /app

COPY requirements.txt .

# Install Python dependencies for upgrading and Diffusion requirements.
RUN apt-get clean && apt update && apt install zip unzip wget curl htop g++ gcc libgl1 libglib2.0-0 -y && \
    rm -rf /var/lib/apt/lists/* && \
    python3 -m pip install --no-cache-dir --upgrade pip && \
    pip3 install --no-cache-dir -r requirements.txt && \
    rm -rf /root/.cache/pip/* && \
    apt-get purge -y --auto-remove

# Copy the application code
COPY . .

# Run the FastAPI server on container startup
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8200", "--reload"]