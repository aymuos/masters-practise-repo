# base Python image
FROM python:3.13-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \ 
     build-essential pkg-config python3-dev default-jre-headless && rm -rf /var/lib/apt/lists/*


# Set JAVA_HOME environment variable for PySpark
ENV JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64
ENV PATH="$JAVA_HOME/bin:$PATH"

RUN python -m pip install --upgrade pip setuptools

# Copy requirements (include FastAPI, uvicorn, mlflow, pyspark)
COPY requirements.txt .

RUN pip install --no-cache-dir --prefer-binary -r requirements.txt -v

# Copy app source
COPY src/ ./src/
# COPY deployment/ ./deployment/
# COPY utils/ ./utils/

# Set environment variables
ENV MLFLOW_TRACKING_URI=http://host.docker.internal:5000
ENV MODEL_NAME=TitanicModel
ENV MODEL_STAGE=Production

# Expose FastAPI port
EXPOSE 8000

# Run FastAPI app with uvicorn
CMD ["uvicorn", "deployment.api:app", "--host", "0.0.0.0", "--port", "8000"]
