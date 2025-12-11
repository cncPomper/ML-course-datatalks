ECR_URL=279586433692.dkr.ecr.eu-north-1.amazonaws.com

REPO_URL=${ECR_URL}/image-prediction-lambda

REMOTE_IMAGE_TAG="${REPO_URL}:v1"

LOCAL_IMAGE_NAME=model-2025-hairstyle

aws configure set region eu-north-1

aws ecr get-login-password \
  --region "eu-north-1" \
| docker login \
  --username AWS \
  --password-stdin ${ECR_URL}

docker build -t ${LOCAL_IMAGE_NAME} .

docker tag ${LOCAL_IMAGE_NAME} ${REMOTE_IMAGE_TAG}

docker push ${REMOTE_IMAGE_TAG}

echo "Pushed image to ${REMOTE_IMAGE_TAG}"
