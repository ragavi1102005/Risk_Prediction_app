# Use the same Python version you trained with
FROM python:3.12-slim

# 1. Install system dependencies: git and git-lfs
RUN apt-get update && apt-get install -y git-lfs \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 2. Copy requirements first to cache them
COPY requirements.txt .

# 3. Install python packages
RUN pip install --no-cache-dir -r requirements.txt

# 4. Copy your *entire* repository
# This includes the .git folder, which is needed for lfs pull
COPY . .

# 5. This is the FIX: Pull the real .pkl files
RUN git-lfs pull

# Expose the port Render expects
EXPOSE 10000

# Command to run the app
CMD gunicorn app:app --bind 0.0.0.0:$PORT