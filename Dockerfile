FROM python:3.10-slim

# Install minimal system deps (useful for DICOM + some pip packages)
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
    && rm -rf /var/lib/apt/lists/*

# Set workdir inside the container
WORKDIR /app

# Copy requirements and install
COPY requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy your source code
COPY src/ ./src/

# Default volumes and command (can be overridden at runtime)
VOLUME ["/data"]

# Default: download 20 cases, no forced redownload
ENTRYPOINT ["python", "-m", "src.download.cmmd_api_download"]
CMD ["--data_dir", "/data", "--max_cases", "20"]
