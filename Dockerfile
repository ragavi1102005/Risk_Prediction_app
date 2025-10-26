# Use Python 3.10 as the base
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy all files
COPY . /app

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose the port Render will use
EXPOSE 10000

# Command to run the app
CMD gunicorn app:app --bind 0.0.0.0:$PORT
