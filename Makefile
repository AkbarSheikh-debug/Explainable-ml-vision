.PHONY: help build run stop push test clean

# Variables
IMAGE_NAME = explainable-ml-vision
VERSION = 1.0.0
DOCKER_USERNAME = your_username  # Replace with your Docker Hub username

# Default target
help:
	@echo "Available commands:"
	@echo "  make build        - Build Docker image"
	@echo "  make run          - Run Docker container"
	@echo "  make stop         - Stop Docker container"
	@echo "  make push         - Push image to Docker Hub"
	@echo "  make test         - Run tests"
	@echo "  make clean        - Remove containers and images"
	@echo "  make logs         - View container logs"
	@echo "  make shell        - Open shell in container"
	@echo "  make compose-up   - Start with docker-compose"
	@echo "  make compose-down - Stop docker-compose services"

# Build Docker image
build:
	docker build -t $(IMAGE_NAME):$(VERSION) .
	docker tag $(IMAGE_NAME):$(VERSION) $(IMAGE_NAME):latest
	@echo "Image built successfully!"

# Run Docker container
run:
	docker run -d \
		--name $(IMAGE_NAME) \
		-p 8501:8501 \
		-v $(PWD)/data:/app/data \
		-v $(PWD)/models:/app/models \
		-v $(PWD)/logs:/app/logs \
		$(IMAGE_NAME):latest
	@echo "Container started. Open http://localhost:8501"

# Stop Docker container
stop:
	docker stop $(IMAGE_NAME) || true
	docker rm $(IMAGE_NAME) || true
	@echo "Container stopped and removed."

# Push to Docker Hub
push:
	docker tag $(IMAGE_NAME):$(VERSION) $(DOCKER_USERNAME)/$(IMAGE_NAME):$(VERSION)
	docker tag $(IMAGE_NAME):latest $(DOCKER_USERNAME)/$(IMAGE_NAME):latest
	docker push $(DOCKER_USERNAME)/$(IMAGE_NAME):$(VERSION)
	docker push $(DOCKER_USERNAME)/$(IMAGE_NAME):latest
	@echo "Images pushed to Docker Hub!"

# Run tests
test:
	@echo "Running tests..."
	docker run --rm $(IMAGE_NAME):latest python -c "import torch; print('PyTorch:', torch.__version__)"
	docker run --rm $(IMAGE_NAME):latest python -c "import streamlit; print('Streamlit:', streamlit.__version__)"
	@echo "All tests passed!"

# Clean up
clean:
	docker stop $(IMAGE_NAME) 2>/dev/null || true
	docker rm $(IMAGE_NAME) 2>/dev/null || true
	docker rmi $(IMAGE_NAME):$(VERSION) 2>/dev/null || true
	docker rmi $(IMAGE_NAME):latest 2>/dev/null || true
	docker system prune -f
	@echo "Cleanup completed!"

# View logs
logs:
	docker logs -f $(IMAGE_NAME)

# Open shell in container
shell:
	docker exec -it $(IMAGE_NAME) /bin/bash

# Docker Compose commands
compose-up:
	docker-compose up -d
	@echo "Services started with docker-compose"

compose-down:
	docker-compose down
	@echo "Services stopped with docker-compose"

compose-logs:
	docker-compose logs -f