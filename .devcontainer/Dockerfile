# Use the official Python 3.12 slim image as the base
FROM python:3.12-slim

# Set the working directory
WORKDIR /workspace

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    curl \
    nano \
    && rm -rf /var/lib/apt/lists/*

# Update pip
RUN pip install --upgrade pip --root-user-action=ignore

# Install Poetry
RUN pip install poetry --root-user-action=ignore

# Copy the workspace
COPY . /workspace

# Set an entrypoint or default command
CMD ["bash"]
