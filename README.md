# 🤖 ORYZA AI BOX - Professional Multi-Platform AI System

[![Python](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1+-red.svg)](https://pytorch.org/)
[![Docker](https://img.shields.io/badge/docker-ready-blue.svg)](https://www.docker.com/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

## 🎯 **Tổng quan**

**Oryza AI Box** là hệ thống AI chuyên nghiệp chạy trên đa nền tảng, tập trung vào phân tích con người và phương tiện giao thông. Hệ thống được thiết kế để triển khai trên các thiết bị edge computing với hiệu năng cao.

### ✨ **Tính năng chính**

#### 👥 **Human Analysis**
- **Face Recognition:** Nhận dạng khuôn mặt với độ chính xác cao
- **Person Detection:** Phát hiện và đếm người trong thời gian thực
- **Behavior Analysis:** Phân tích hành vi và hoạt động
- **Age & Gender:** Ước tính tuổi và giới tính
- **Emotion Recognition:** Nhận dạng cảm xúc

#### 🚗 **Vehicle Analysis**
- **License Plate Recognition:** Nhận dạng biển số xe
- **Vehicle Classification:** Phân loại loại xe (ô tô, xe máy, xe tải, xe buýt)
- **Brand & Model Recognition:** Nhận dạng thương hiệu và mẫu xe
- **Traffic Analytics:** Phân tích lưu lượng giao thông
- **Speed Estimation:** Ước tính tốc độ xe

### 🏗️ **Kiến trúc hệ thống**

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Development   │    │   CI/CD         │    │   Edge Devices  │
│   (myai)        │───▶│   Pipeline      │───▶│   (Pi5, Rock5)  │
│   - Training    │    │   - Build       │    │   - Inference   │
│   - Testing     │    │   - Deploy      │    │   - Real-time   │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

### 🖥️ **Supported Platforms**

| Platform | CPU | RAM | AI Accelerator | Performance |
|----------|-----|-----|----------------|-------------|
| **Development (myai)** | Intel Xeon 88 cores | 125GB | RTX 4060 Ti 16GB | Training & Development |
| **Raspberry Pi 5** | ARM Cortex-A76 4 cores | 8GB | Hailo-8 (13 TOPS) | High-performance Edge |
| **Radxa Rock 5 ITX** | ARM Cortex-A76 6 cores | 8GB | NPU (6 TOPS) | Medium-performance Edge |
| **Jetson Nano** | ARM Cortex-A57 4 cores | 4GB | Maxwell GPU (472 GFLOPS) | Entry-level Edge |
| **Core i5 Machine** | Intel Core i5 | 32GB | 12GB GPU | Testing & Validation |

## 🚀 **Quick Start**

### 📋 **Prerequisites**

- Python 3.9+
- CUDA 12.0+ (for GPU acceleration)
- Docker & Docker Compose
- Git

### 🛠️ **Installation**

1. **Clone repository:**
```bash
git clone https://github.com/oryza-ai/ai-box.git
cd ai-box
```

2. **Setup virtual environment:**
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate     # Windows
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt

# For GPU support
pip install -e .[gpu]

# For edge devices
pip install -e .[edge]

# For OCR features
pip install -e .[ocr]
```

4. **Run development server:**
```bash
python -m oryza_ai_box.server
```

### 🐳 **Docker Deployment**

```bash
# Build image
docker build -t oryza-ai-box:latest .

# Run container
docker run -d \
  --name ai-box \
  --gpus all \
  -p 8000:8000 \
  -v $(pwd)/data:/app/data \
  oryza-ai-box:latest
```

## 📁 **Project Structure**

```
oryza_ai_box/
├── ai_models/                 # AI Models
│   ├── human_analysis/        # Face recognition, person detection
│   ├── vehicle_analysis/      # License plate, vehicle classification
│   └── common/               # Shared utilities
├── services/                 # Microservices
│   ├── api_gateway/          # API Gateway
│   ├── model_server/         # Model serving
│   └── data_manager/         # Data management
├── deployment/               # Deployment configs
│   ├── docker/              # Docker files
│   ├── k8s/                 # Kubernetes manifests
│   └── scripts/             # Deployment scripts
├── config/                  # Configuration files
├── tests/                   # Test cases
├── docs/                    # Documentation
└── tools/                   # Development tools
```

## 🔧 **Configuration**

### 🌍 **Environment Variables**

```bash
# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
API_WORKERS=4

# Database
DATABASE_URL=postgresql://user:pass@localhost/aibox
REDIS_URL=redis://localhost:6379

# AI Models
MODEL_PATH=/app/models
DEVICE=cuda  # or cpu, mps

# Monitoring
PROMETHEUS_PORT=9090
LOG_LEVEL=INFO
```

### ⚙️ **Device-specific Configuration**

#### Raspberry Pi 5 + Hailo-8
```yaml
platform: raspberry_pi_5
accelerator: hailo_8
performance_mode: high
models:
  - face_recognition
  - person_detection
```

#### Radxa Rock 5 ITX
```yaml
platform: radxa_rock_5
accelerator: npu
performance_mode: medium
models:
  - vehicle_classification
  - license_plate_recognition
```

## 📊 **Performance Benchmarks**

### 🎯 **Target Performance**

| Model | Platform | Resolution | FPS | Accuracy |
|-------|----------|------------|-----|----------|
| Face Recognition | Pi 5 + Hailo | 1080p | 30+ | 95%+ |
| Person Detection | Pi 5 + Hailo | 1080p | 30+ | 90%+ mAP |
| Vehicle Classification | Rock 5 ITX | 720p | 25+ | 90%+ |
| License Plate OCR | Rock 5 ITX | 720p | 25+ | 98%+ |

## 🧪 **Testing**

```bash
# Run all tests
pytest

# Run specific test categories
pytest -m unit          # Unit tests
pytest -m integration   # Integration tests
pytest -m gpu          # GPU tests
pytest -m edge         # Edge device tests

# Run with coverage
pytest --cov=oryza_ai_box --cov-report=html
```

## 📚 **Documentation**

- [API Documentation](docs/api.md)
- [Model Documentation](docs/models.md)
- [Deployment Guide](docs/deployment.md)
- [Development Guide](docs/development.md)
- [Troubleshooting](docs/troubleshooting.md)

## 🤝 **Contributing**

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

### 🔧 **Development Setup**

```bash
# Install development dependencies
pip install -e .[dev]

# Setup pre-commit hooks
pre-commit install

# Run code formatting
black .
flake8 .
mypy .
```

## 📄 **License**

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 **Acknowledgments**

- [PyTorch](https://pytorch.org/) - Deep learning framework
- [OpenCV](https://opencv.org/) - Computer vision library
- [FastAPI](https://fastapi.tiangolo.com/) - Modern web framework
- [Ultralytics](https://ultralytics.com/) - YOLO models
- [MediaPipe](https://mediapipe.dev/) - ML solutions

## 📞 **Support**

- 📧 Email: support@oryza.vn
- 🐛 Issues: [GitHub Issues](https://github.com/oryza-ai/ai-box/issues)
- 📖 Documentation: [docs.oryza-ai.com](https://docs.oryza-ai.com)
- 💬 Community: [Discord](https://discord.gg/oryza-ai)

---

**Made with ❤️ by Oryza AI Team**
