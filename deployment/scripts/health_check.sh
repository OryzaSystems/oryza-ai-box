#!/bin/bash

# ==========================================
# AI Box - Health Check Script
# Universal health check for all services
# ==========================================

set -e

# Configuration
SERVICE_NAME=${SERVICE_NAME:-"unknown"}
HEALTH_PORT=${HEALTH_PORT:-8000}
HEALTH_ENDPOINT=${HEALTH_ENDPOINT:-"/health"}
TIMEOUT=${TIMEOUT:-10}

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Logging function
log() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')] [HEALTH]${NC} $1"
}

error() {
    echo -e "${RED}[$(date +'%Y-%m-%d %H:%M:%S')] [ERROR]${NC} $1" >&2
}

warning() {
    echo -e "${YELLOW}[$(date +'%Y-%m-%d %H:%M:%S')] [WARNING]${NC} $1"
}

# Health check function
check_health() {
    local url="http://localhost:${HEALTH_PORT}${HEALTH_ENDPOINT}"
    
    log "Checking health for ${SERVICE_NAME} at ${url}"
    
    # Check if port is listening
    if ! nc -z localhost ${HEALTH_PORT} 2>/dev/null; then
        error "Port ${HEALTH_PORT} is not listening"
        return 1
    fi
    
    # HTTP health check
    if command -v curl >/dev/null 2>&1; then
        response=$(curl -s -f -m ${TIMEOUT} "${url}" 2>/dev/null)
        if [ $? -eq 0 ]; then
            log "Health check passed: ${response}"
            return 0
        else
            error "Health check failed: curl returned non-zero exit code"
            return 1
        fi
    elif command -v wget >/dev/null 2>&1; then
        if wget -q -T ${TIMEOUT} -O - "${url}" >/dev/null 2>&1; then
            log "Health check passed (wget)"
            return 0
        else
            error "Health check failed: wget returned non-zero exit code"
            return 1
        fi
    else
        warning "Neither curl nor wget available, using basic port check"
        if nc -z localhost ${HEALTH_PORT} 2>/dev/null; then
            log "Port check passed"
            return 0
        else
            error "Port check failed"
            return 1
        fi
    fi
}

# Process check function
check_process() {
    local process_name=${1:-"python"}
    
    if pgrep -f "${process_name}" >/dev/null 2>&1; then
        log "Process '${process_name}' is running"
        return 0
    else
        error "Process '${process_name}' is not running"
        return 1
    fi
}

# Disk space check
check_disk_space() {
    local threshold=${1:-90}
    local usage=$(df /app 2>/dev/null | awk 'NR==2 {print $5}' | sed 's/%//')
    
    if [ -n "$usage" ] && [ "$usage" -lt "$threshold" ]; then
        log "Disk usage: ${usage}% (OK)"
        return 0
    else
        warning "Disk usage: ${usage}% (High)"
        return 1
    fi
}

# Memory check
check_memory() {
    local threshold=${1:-90}
    
    if command -v free >/dev/null 2>&1; then
        local usage=$(free | awk 'NR==2{printf "%.0f", $3*100/$2}')
        if [ "$usage" -lt "$threshold" ]; then
            log "Memory usage: ${usage}% (OK)"
            return 0
        else
            warning "Memory usage: ${usage}% (High)"
            return 1
        fi
    else
        log "Memory check skipped (free command not available)"
        return 0
    fi
}

# Main health check
main() {
    log "Starting health check for ${SERVICE_NAME}"
    
    local exit_code=0
    
    # HTTP health check (primary)
    if ! check_health; then
        exit_code=1
    fi
    
    # Additional checks (non-critical)
    check_process || true
    check_disk_space || true
    check_memory || true
    
    if [ $exit_code -eq 0 ]; then
        log "All health checks passed"
    else
        error "Health check failed"
    fi
    
    exit $exit_code
}

# Run main function
main "$@"
