#!/bin/bash

# Build script for Explainable ML Vision Docker image

set -e  # Exit on error

# Variables
IMAGE_NAME="explainable-ml-vision"
VERSION="1.0.0"
DOCKER_USERNAME=""  # Set your Docker Hub username
PLATFORMS="linux/amd64,linux/arm64"  # Multi-platform build

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_message() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    print_error "Docker is not running. Please start Docker and try again."
    exit 1
fi

# Build the Docker image
build_image() {
    print_message "Building Docker image: ${IMAGE_NAME}:${VERSION}"
    
    # Standard build
    docker build -t ${IMAGE_NAME}:${VERSION} .
    
    # Also tag as latest
    docker tag ${IMAGE_NAME}:${VERSION} ${IMAGE_NAME}:latest
    
    print_message "Build completed successfully!"
}

# Multi-platform build (requires Docker Buildx)
build_multi_platform() {
    print_message "Building multi-platform image..."
    
    # Create and use builder
    docker buildx create --name mybuilder --use
    
    # Build for multiple platforms
    docker buildx build \
        --platform ${PLATFORMS} \
        -t ${IMAGE_NAME}:${VERSION} \
        -t ${IMAGE_NAME}:latest \
        --push .
    
    print_message "Multi-platform build completed!"
}

# Push to Docker Hub
push_to_dockerhub() {
    if [ -z "$DOCKER_USERNAME" ]; then
        print_warning "Docker Hub username not set. Please edit build.sh to set DOCKER_USERNAME"
        read -p "Enter Docker Hub username: " DOCKER_USERNAME
    fi
    
    # Tag for Docker Hub
    docker tag ${IMAGE_NAME}:${VERSION} ${DOCKER_USERNAME}/${IMAGE_NAME}:${VERSION}
    docker tag ${IMAGE_NAME}:latest ${DOCKER_USERNAME}/${IMAGE_NAME}:latest
    
    # Login to Docker Hub
    print_message "Logging in to Docker Hub..."
    docker login -u ${DOCKER_USERNAME}
    
    # Push images
    print_message "Pushing images to Docker Hub..."
    docker push ${DOCKER_USERNAME}/${IMAGE_NAME}:${VERSION}
    docker push ${DOCKER_USERNAME}/${IMAGE_NAME}:latest
    
    print_message "Images pushed successfully to Docker Hub!"
}

# Run tests
run_tests() {
    print_message "Running container tests..."
    
    # Start container in background
    docker run -d --name test-container -p 8501:8501 ${IMAGE_NAME}:${VERSION}
    
    # Wait for container to start
    sleep 10
    
    # Check if container is running
    if docker ps | grep -q test-container; then
        print_message "Container is running successfully!"
        
        # Test health endpoint
        curl -f http://localhost:8501/_stcore/health || {
            print_error "Health check failed"
            docker logs test-container
            exit 1
        }
        
        print_message "Health check passed!"
    else
        print_error "Container failed to start"
        docker logs test-container
        exit 1
    fi
    
    # Clean up
    docker stop test-container
    docker rm test-container
    print_message "Tests completed successfully!"
}

# Display help
show_help() {
    echo "Usage: ./build.sh [OPTION]"
    echo ""
    echo "Options:"
    echo "  build           Build Docker image"
    echo "  multi-platform  Build for multiple platforms (requires Docker Buildx)"
    echo "  push            Push image to Docker Hub"
    echo "  test            Run tests on the built image"
    echo "  all             Build, test, and push (if logged in)"
    echo "  help            Display this help message"
}

# Main execution
case "$1" in
    "build")
        build_image
        ;;
    "multi-platform")
        build_multi_platform
        ;;
    "push")
        push_to_dockerhub
        ;;
    "test")
        run_tests
        ;;
    "all")
        build_image
        run_tests
        push_to_dockerhub
        ;;
    "help"|"")
        show_help
        ;;
    *)
        print_error "Unknown option: $1"
        show_help
        exit 1
        ;;
esac