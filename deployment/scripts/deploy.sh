#!/bin/bash

# ==========================================
# AI BOX - Deployment Script
# Automated deployment to edge devices
# ==========================================

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
CONFIG_FILE="$PROJECT_ROOT/config/deployment.yml"

# Default values
ENVIRONMENT="production"
PLATFORM=""
TARGET_HOST=""
TARGET_USER="aibox"
SSH_KEY=""
DOCKER_REGISTRY="ghcr.io"
IMAGE_TAG="latest"
BACKUP_BEFORE_DEPLOY=true
HEALTH_CHECK_TIMEOUT=300

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

usage() {
    cat << EOF
🚀 AI Box Deployment Script

Usage: $0 [OPTIONS]

Options:
    -e, --environment ENV     Deployment environment (dev/staging/production)
    -p, --platform PLATFORM  Target platform (raspberry-pi-5/radxa-rock-5/jetson-nano/core-i5)
    -h, --host HOST          Target host IP address
    -u, --user USER          SSH user (default: aibox)
    -k, --key PATH           SSH private key path
    -t, --tag TAG            Docker image tag (default: latest)
    -r, --registry URL       Docker registry URL (default: ghcr.io)
    --no-backup              Skip backup before deployment
    --help                   Show this help message

Examples:
    # Deploy to Raspberry Pi 5
    $0 -e production -p raspberry-pi-5 -h 192.168.1.100 -k ~/.ssh/id_rsa

    # Deploy to Radxa Rock 5 ITX
    $0 -e production -p radxa-rock-5 -h 192.168.1.101 -u rock -k ~/.ssh/rock_key

    # Deploy to Jetson Nano
    $0 -e production -p jetson-nano -h 192.168.1.102

EOF
}

# ==========================================
# Parse Command Line Arguments
# ==========================================

while [[ $# -gt 0 ]]; do
    case $1 in
        -e|--environment)
            ENVIRONMENT="$2"
            shift 2
            ;;
        -p|--platform)
            PLATFORM="$2"
            shift 2
            ;;
        -h|--host)
            TARGET_HOST="$2"
            shift 2
            ;;
        -u|--user)
            TARGET_USER="$2"
            shift 2
            ;;
        -k|--key)
            SSH_KEY="$2"
            shift 2
            ;;
        -t|--tag)
            IMAGE_TAG="$2"
            shift 2
            ;;
        -r|--registry)
            DOCKER_REGISTRY="$2"
            shift 2
            ;;
        --no-backup)
            BACKUP_BEFORE_DEPLOY=false
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

# ==========================================
# Validation
# ==========================================

validate_inputs() {
    log "🔍 Validating deployment parameters..."
    
    if [[ -z "$PLATFORM" ]]; then
        error "Platform is required. Use -p or --platform option."
    fi
    
    if [[ -z "$TARGET_HOST" ]]; then
        error "Target host is required. Use -h or --host option."
    fi
    
    if [[ -n "$SSH_KEY" && ! -f "$SSH_KEY" ]]; then
        error "SSH key file not found: $SSH_KEY"
    fi
    
    # Validate platform
    case $PLATFORM in
        raspberry-pi-5|radxa-rock-5|jetson-nano|core-i5)
            ;;
        *)
            error "Invalid platform: $PLATFORM. Supported: raspberry-pi-5, radxa-rock-5, jetson-nano, core-i5"
            ;;
    esac
    
    success "Validation passed"
}

# ==========================================
# SSH Connection Test
# ==========================================

test_ssh_connection() {
    log "🔗 Testing SSH connection to $TARGET_USER@$TARGET_HOST..."
    
    SSH_CMD="ssh"
    if [[ -n "$SSH_KEY" ]]; then
        SSH_CMD="ssh -i $SSH_KEY"
    fi
    
    if $SSH_CMD -o ConnectTimeout=10 -o BatchMode=yes "$TARGET_USER@$TARGET_HOST" "echo 'SSH connection successful'" > /dev/null 2>&1; then
        success "SSH connection established"
    else
        error "Cannot establish SSH connection to $TARGET_USER@$TARGET_HOST"
    fi
}

# ==========================================
# Platform Detection
# ==========================================

detect_platform() {
    log "🔍 Detecting platform on target device..."
    
    REMOTE_PLATFORM=$($SSH_CMD "$TARGET_USER@$TARGET_HOST" "
        if [[ -f /proc/device-tree/model ]]; then
            MODEL=\$(cat /proc/device-tree/model 2>/dev/null | tr -d '\0')
            if [[ \$MODEL == *'Raspberry Pi 5'* ]]; then
                echo 'raspberry-pi-5'
            elif [[ \$MODEL == *'Rock 5'* ]]; then
                echo 'radxa-rock-5'
            elif [[ \$MODEL == *'Jetson'* ]]; then
                echo 'jetson-nano'
            else
                echo 'unknown'
            fi
        else
            # x86_64 systems
            if lscpu | grep -q 'Intel'; then
                echo 'core-i5'
            else
                echo 'unknown'
            fi
        fi
    ")
    
    if [[ "$REMOTE_PLATFORM" != "$PLATFORM" && "$REMOTE_PLATFORM" != "unknown" ]]; then
        warning "Platform mismatch: specified '$PLATFORM', detected '$REMOTE_PLATFORM'"
        read -p "Continue anyway? (y/N): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            error "Deployment cancelled"
        fi
    fi
    
    success "Platform detection completed"
}

# ==========================================
# Backup Current Deployment
# ==========================================

backup_current_deployment() {
    if [[ "$BACKUP_BEFORE_DEPLOY" == "true" ]]; then
        log "💾 Creating backup of current deployment..."
        
        BACKUP_NAME="aibox-backup-$(date +%Y%m%d-%H%M%S)"
        
        $SSH_CMD "$TARGET_USER@$TARGET_HOST" "
            if [[ -d /opt/oryza-ai-box ]]; then
                sudo mkdir -p /opt/backups
                sudo tar -czf /opt/backups/$BACKUP_NAME.tar.gz -C /opt oryza-ai-box
                echo 'Backup created: /opt/backups/$BACKUP_NAME.tar.gz'
            else
                echo 'No existing deployment found, skipping backup'
            fi
        "
        
        success "Backup completed"
    fi
}

# ==========================================
# Deploy Docker Compose
# ==========================================

deploy_docker_compose() {
    log "🐳 Deploying Docker Compose configuration..."
    
    # Copy docker-compose files
    scp -i "$SSH_KEY" "$PROJECT_ROOT/docker-compose.yml" "$TARGET_USER@$TARGET_HOST:/tmp/"
    scp -i "$SSH_KEY" "$PROJECT_ROOT/deployment/docker-compose.$PLATFORM.yml" "$TARGET_USER@$TARGET_HOST:/tmp/" 2>/dev/null || true
    
    # Copy environment configuration
    scp -i "$SSH_KEY" "$PROJECT_ROOT/config/env.$PLATFORM" "$TARGET_USER@$TARGET_HOST:/tmp/.env" 2>/dev/null || \
    scp -i "$SSH_KEY" "$PROJECT_ROOT/config/env.example" "$TARGET_USER@$TARGET_HOST:/tmp/.env"
    
    $SSH_CMD "$TARGET_USER@$TARGET_HOST" "
        sudo mkdir -p /opt/oryza-ai-box
        sudo mv /tmp/docker-compose.yml /opt/oryza-ai-box/
        sudo mv /tmp/.env /opt/oryza-ai-box/
        
        if [[ -f /tmp/docker-compose.$PLATFORM.yml ]]; then
            sudo mv /tmp/docker-compose.$PLATFORM.yml /opt/oryza-ai-box/
        fi
        
        cd /opt/oryza-ai-box
        
        # Set platform-specific environment
        echo 'AI_PLATFORM=$PLATFORM' | sudo tee -a .env
        echo 'ENVIRONMENT=$ENVIRONMENT' | sudo tee -a .env
        echo 'IMAGE_TAG=$IMAGE_TAG' | sudo tee -a .env
        
        # Pull latest images
        sudo docker-compose pull
        
        # Deploy services
        if [[ -f docker-compose.$PLATFORM.yml ]]; then
            sudo docker-compose -f docker-compose.yml -f docker-compose.$PLATFORM.yml up -d
        else
            sudo docker-compose up -d
        fi
    "
    
    success "Docker Compose deployment completed"
}

# ==========================================
# Health Check
# ==========================================

health_check() {
    log "🏥 Performing health check..."
    
    local start_time=$(date +%s)
    local timeout=$HEALTH_CHECK_TIMEOUT
    
    while true; do
        local current_time=$(date +%s)
        local elapsed=$((current_time - start_time))
        
        if [[ $elapsed -gt $timeout ]]; then
            error "Health check timeout after ${timeout}s"
        fi
        
        if curl -f -s "http://$TARGET_HOST:8000/health" > /dev/null 2>&1; then
            success "Health check passed"
            break
        fi
        
        log "Waiting for services to start... (${elapsed}s/${timeout}s)"
        sleep 10
    done
}

# ==========================================
# Post-deployment Tasks
# ==========================================

post_deployment() {
    log "🔧 Running post-deployment tasks..."
    
    $SSH_CMD "$TARGET_USER@$TARGET_HOST" "
        cd /opt/oryza-ai-box
        
        # Clean up old images
        sudo docker system prune -f
        
        # Setup log rotation
        sudo mkdir -p /etc/logrotate.d
        echo '/opt/oryza-ai-box/logs/*.log {
            daily
            rotate 7
            compress
            delaycompress
            missingok
            notifempty
            create 644 $TARGET_USER $TARGET_USER
        }' | sudo tee /etc/logrotate.d/aibox
        
        # Setup systemd service for auto-start
        echo '[Unit]
Description=AI Box Services
Requires=docker.service
After=docker.service

[Service]
Type=oneshot
RemainAfterExit=yes
WorkingDirectory=/opt/oryza-ai-box
ExecStart=/usr/bin/docker-compose up -d
ExecStop=/usr/bin/docker-compose down
TimeoutStartSec=0

[Install]
WantedBy=multi-user.target' | sudo tee /etc/systemd/system/aibox.service
        
        sudo systemctl daemon-reload
        sudo systemctl enable aibox.service
    "
    
    success "Post-deployment tasks completed"
}

# ==========================================
# Main Deployment Flow
# ==========================================

main() {
    echo "🚀 AI Box Deployment Script"
    echo "=================================="
    echo "Environment: $ENVIRONMENT"
    echo "Platform: $PLATFORM"
    echo "Target: $TARGET_USER@$TARGET_HOST"
    echo "Image Tag: $IMAGE_TAG"
    echo "=================================="
    echo
    
    validate_inputs
    test_ssh_connection
    detect_platform
    backup_current_deployment
    deploy_docker_compose
    health_check
    post_deployment
    
    echo
    success "🎉 Deployment completed successfully!"
    echo
    echo "📊 Service URLs:"
    echo "  • API Gateway: http://$TARGET_HOST:8000"
    echo "  • Model Server: http://$TARGET_HOST:8001"
    echo "  • Data Manager: http://$TARGET_HOST:8002"
    echo "  • Grafana: http://$TARGET_HOST:3000"
    echo "  • Prometheus: http://$TARGET_HOST:9090"
    echo
    echo "📝 Logs: /opt/oryza-ai-box/logs/"
    echo "💾 Backups: /opt/backups/"
    echo
}

# Run main function
main "$@"
