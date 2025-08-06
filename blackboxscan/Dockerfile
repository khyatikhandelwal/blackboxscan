FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy code
COPY . /app

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose port (must match your Flask app)
EXPOSE 5000

# Run the Flask app
CMD ["python", "app.py"]