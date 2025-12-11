FROM agrigorev/model-2025-hairstyle:v1

ARG MODEL_NAME=hair_classifier_empty.onnx
ENV MODEL_NAME=${MODEL_NAME}

RUN pip install onnxruntime numpy pillow

COPY lambda_function.py .

CMD ["lambda_function.lambda_handler"]