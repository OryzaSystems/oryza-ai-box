#!/bin/bash

# ==========================================
# AI Box - Multi-Platform Docker Build Script
# Build and push Docker images for all platforms
# ==========================================

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
REGISTRY=${REGISTRY:-"ghcr.io/oryzasystems"}
PROJECT_NAME="ai-box"
IMAGE_TAG=${IMAGE_TAG:-"latest"}
PLATFORMS="linux/amd64,linux/arm64"
PUSH=${PUSH:-"false"}

# Services to build
SERVICES=("base" "api-gateway" "model-server" "data-manager")

# ==========================================
# Helper Functions
# ==========================================

log() {
    echo -e "${BLUE}[$(date +'%Y-%m-%d %H:%M:%S')]${NC} $1"
}

success() {
    echo -e "${GREEN}✅ $1${NC}"
}

warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

error() {
    echo -e "${RED}❌ $1${NC}"
    exit 1
}

# ==========================================
# Docker Buildx Setup
# ==========================================

setup_buildx() {
    log "Setting up Docker Buildx for multi-platform builds..."
    
    # Create buildx builder if it doesn't exist
    if ! docker buildx ls | grep -q "aibox-builder"; then
        log "Creating new buildx builder: aibox-builder"
        docker buildx create --name aibox-builder --driver docker-container --bootstrap
    fi
    
    # Use the builder
    docker buildx use aibox-builder
    
    # Inspect builder
    docker buildx inspect --bootstrap
    
    success "Docker Buildx setup completed"
}

# ==========================================
# Build Functions
# ==========================================

build_base_image() {
    log "Building base image..."
    
    local image_name="${REGISTRY}/${PROJECT_NAME}-base:${IMAGE_TAG}"
    
    docker buildx build \
        --platform ${PLATFORMS} \
        --file deployment/docker/Dockerfile.base \
        --tag ${image_name} \
        ${PUSH:+--push} \
        --progress=plain \
        .
    
    success "Base image built: ${image_name}"
}

build_service_image() {
    local service=$1
    log "Building ${service} image..."
    
    local image_name="${REGISTRY}/${PROJECT_NAME}-${service}:${IMAGE_TAG}"
    local dockerfile="deployment/docker/Dockerfile.${service}"
    
    if [[ ! -f ${dockerfile} ]]; then
        error "Dockerfile not found: ${dockerfile}"
    fi
    
    # Build arguments
    local build_args=""
    case ${service} in
        "model-server")
            build_args="--build-arg PLATFORM=auto"
            ;;
    esac
    
    docker buildx build \
        --platform ${PLATFORMS} \
        --file ${dockerfile} \
        --tag ${image_name} \
        ${build_args} \
        ${PUSH:+--push} \
        --progress=plain \
        .
    
    success "${service} image built: ${image_name}"
}

# ==========================================
# Platform-Specific Builds
# ==========================================

build_platform_specific() {
    local platform=$1
    log "Building platform-specific images for: ${platform}"
    
    # Model server with platform-specific optimizations
    local platforms_map=(
        "raspberry-pi-5:linux/arm64"
        "radxa-rock-5:linux/arm64"
        "jetson-nano:linux/arm64"
        "core-i5:linux/amd64"
    )
    
    for platform_config in "${platforms_map[@]}"; do
        IFS=':' read -r platform_name platform_arch <<< "${platform_config}"
        
        if [[ ${platform} == "all" || ${platform} == ${platform_name} ]]; then
            log "Building model-server for ${platform_name} (${platform_arch})..."
            
            local image_name="${REGISTRY}/${PROJECT_NAME}-model-server:${platform_name}-${IMAGE_TAG}"
            
            docker buildx build \
                --platform ${platform_arch} \
                --file deployment/docker/Dockerfile.model-server \
                --target ${platform_name} \
                --tag ${image_name} \
                --build-arg PLATFORM=${platform_name} \
                ${PUSH:+--push} \
                --progress=plain \
                .
            
            success "Platform-specific image built: ${image_name}"
        fi
    done
}

# ==========================================
# Image Testing
# ==========================================

test_images() {
    log "Testing built images..."
    
    for service in "${SERVICES[@]}"; do
        if [[ ${service} == "base" ]]; then
            continue
        fi
        
        local image_name="${REGISTRY}/${PROJECT_NAME}-${service}:${IMAGE_TAG}"
        
        log "Testing ${service} image..."
        
        # Test image can run
        if docker run --rm --platform linux/amd64 ${image_name} python --version >/dev/null 2>&1; then
            success "${service} image test passed"
        else
            error "${service} image test failed"
        fi
    done
}

# ==========================================
# Cleanup
# ==========================================

cleanup() {
    log "Cleaning up build cache..."
    docker buildx prune -f
    success "Cleanup completed"
}

# ==========================================
# Main Build Process
# ==========================================

main() {
    echo "🐳 AI Box Multi-Platform Docker Build"
    echo "======================================"
    echo "Registry: ${REGISTRY}"
    echo "Tag: ${IMAGE_TAG}"
    echo "Platforms: ${PLATFORMS}"
    echo "Push: ${PUSH}"
    echo "======================================"
    echo
    
    # Setup
    setup_buildx
    
    # Build base image first
    build_base_image
    
    # Build service images
    for service in "${SERVICES[@]}"; do
        if [[ ${service} != "base" ]]; then
            build_service_image ${service}
        fi
    done
    
    # Build platform-specific images
    if [[ ${BUILD_PLATFORM_SPECIFIC:-"false"} == "true" ]]; then
        build_platform_specific ${PLATFORM_TARGET:-"all"}
    fi
    
    # Test images (only if not pushing)
    if [[ ${PUSH} != "true" ]]; then
        test_images
    fi
    
    # Cleanup
    cleanup
    
    echo
    success "🎉 All images built successfully!"
    echo
    echo "📋 Built images:"
    for service in "${SERVICES[@]}"; do
        echo "  • ${REGISTRY}/${PROJECT_NAME}-${service}:${IMAGE_TAG}"
    done
    echo
    
    if [[ ${PUSH} == "true" ]]; then
        success "Images pushed to registry: ${REGISTRY}"
    else
        warning "Images built locally. Use PUSH=true to push to registry."
    fi
}

# ==========================================
# Command Line Interface
# ==========================================

usage() {
    cat << EOF
🐳 AI Box Multi-Platform Docker Build Script

Usage: $0 [OPTIONS]

Options:
    -r, --registry REGISTRY     Docker registry (default: ghcr.io/oryzasystems)
    -t, --tag TAG              Image tag (default: latest)
    -p, --platforms PLATFORMS  Target platforms (default: linux/amd64,linux/arm64)
    --push                     Push images to registry
    --platform-specific        Build platform-specific images
    --platform-target TARGET   Platform target for specific builds (raspberry-pi-5, radxa-rock-5, jetson-nano, core-i5, all)
    --no-cache                 Build without cache
    --help                     Show this help message

Examples:
    # Build all images locally
    $0

    # Build and push to registry
    $0 --push

    # Build for specific platforms
    $0 --platforms linux/arm64 --push

    # Build platform-specific images
    $0 --platform-specific --platform-target raspberry-pi-5 --push

    # Build with custom registry and tag
    $0 --registry myregistry.com/aibox --tag v1.0.0 --push

EOF
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -r|--registry)
            REGISTRY="$2"
            shift 2
            ;;
        -t|--tag)
            IMAGE_TAG="$2"
            shift 2
            ;;
        -p|--platforms)
            PLATFORMS="$2"
            shift 2
            ;;
        --push)
            PUSH="true"
            shift
            ;;
        --platform-specific)
            BUILD_PLATFORM_SPECIFIC="true"
            shift
            ;;
        --platform-target)
            PLATFORM_TARGET="$2"
            shift 2
            ;;
        --no-cache)
            export DOCKER_BUILDKIT_NO_CACHE=1
            shift
            ;;
        --help)
            usage
            exit 0
            ;;
        *)
            error "Unknown option: $1"
            ;;
    esac
done

# Run main function
main "$@"
