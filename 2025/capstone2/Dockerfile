# Use Python 3.13 slim image as base
FROM python:3.13-slim

ARG MODEL_NAME=model.onnx
ENV MODEL_NAME=${MODEL_NAME}

# Install uv
RUN pip install onnxruntime numpy pandas scikit-learn

# Copy application files
COPY predict.py lambda_function.py .

# Expose port if needed (e.g., for Flask/FastAPI)
# EXPOSE 8000

# Default command - can be overridden
CMD ["lambda_function.handler"]
