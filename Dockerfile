FROM python:3.10

WORKDIR /app

# Install system deps (needed for SimpleITK)
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy project
COPY . .

# Expose Flask port
EXPOSE 5000

# Run app
CMD ["python", "src/inference.py"]