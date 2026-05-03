FROM python:3.11-slim

WORKDIR /app

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Expose port
EXPOSE 8502

# Start Streamlit
CMD ["streamlit", "run", "streamlit_app.py", "--server.port=8502", "--server.address=0.0.0.0"]
