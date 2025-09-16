# GPU-ready serverless base image
FROM runpod/serverless:gpu

WORKDIR /app

# Snellere/zuivere pip
RUN python -m pip install --upgrade pip

# Dependencies
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Code
COPY . /app

# Start (handler.py roept runpod.serverless.start(...) aan)
CMD ["python", "-u", "handler.py"]
