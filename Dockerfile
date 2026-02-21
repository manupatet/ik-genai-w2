FROM python:3.12-slim

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends curl git g++ procps npm && rm -rf /var/lib/apt/lists/*

# Install uv (Python package manager by Astral)
RUN curl -LsSf https://astral.sh/uv/install.sh | sh

# Make uv globally accessible
ENV PATH="/root/.local/bin:/root/.cargo/bin:${PATH}"

EXPOSE 8080

# Set working directory
WORKDIR /workspace
COPY . /workspace

CMD [ "bash" ]
