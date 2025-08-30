# 🤖 AI DEVICE MANAGEMENT - ORYZA AI BOX PROJECT

## 📋 Tổng quan dự án
**Dự án:** Phát triển AI Box với các model AI phân tích con người và phương tiện giao thông
**Mục tiêu:** Triển khai AI models trên multiple platforms với CI/CD pipeline

---

## 🖥️ **MAIN DEVELOPMENT MACHINE - myai**

### Thông tin hệ thống
- **Hostname:** orza-ai
- **OS:** Ubuntu 24.04.1 LTS (Linux 6.14.0-28-generic)
- **Architecture:** x86_64 (64-bit)
- **User:** myai
- **Working Directory:** /home/myai/oryza_ai_box

### CPU - Intel Xeon E5-2699 v4
- **Model:** Intel(R) Xeon(R) CPU E5-2699 v4 @ 2.20GHz
- **Sockets:** 2
- **Cores per socket:** 22
- **Threads per core:** 2
- **Total CPU cores:** 88 (44 physical + 44 logical)
- **Max frequency:** 3.6 GHz
- **Min frequency:** 1.2 GHz
- **Cache:** L1d 1.4MB, L1i 1.4MB, L2 11MB, L3 110MB

### RAM & Storage
- **RAM:** 125GB (Used: 13GB, Available: 112GB)
- **Swap:** 8GB (không sử dụng)
- **Main drive:** NVMe SSD 915GB (Used: 67GB, Available: 802GB)
- **Boot partition:** 1.1GB EFI

### GPU - NVIDIA GeForce RTX 4060 Ti
- **Model:** NVIDIA GeForce RTX 4060 Ti 16GB
- **Driver:** 575.64.03
- **CUDA:** 12.9
- **Memory:** 16GB GDDR6
- **Current usage:** 1GB/16GB
- **Temperature:** 46°C
- **Power:** 13W/165W

### Software Stack
- **Python:** 3.12.3
- **Node.js:** v20.19.4
- **IDE:** Cursor AI
- **Virtualization:** VT-x enabled

### Vai trò trong dự án
- **Primary Development Machine**
- **AI Model Training & Development**
- **CI/CD Pipeline Management**
- **Docker Image Building**
- **GitLab/GitHub Actions Host**

---

## 🍓 **RASPBERRY PI 5 - AI EDGE DEVICE**

### Thông tin cơ bản
- **Model:** Raspberry Pi 5
- **RAM:** 8GB
- **Storage:** MicroSD Card (cần xác định dung lượng)
- **OS:** Raspberry Pi OS (cần xác định version)

### AI Module - Hailo-8
- **Model:** Hailo-8 AI Accelerator
- **Performance:** 13 TOPS (Trillion Operations Per Second)
- **Memory:** 16GB LPDDR4
- **Power:** 2.5W typical
- **Interface:** M.2 PCIe Gen3 x4

### Vai trò trong dự án
- **High-Performance AI Inference**
- **Real-time Video Processing**
- **Edge AI Deployment**
- **Model:** Face Recognition, Person Detection, Behavior Analysis

---

## 🪨 **RADXA ROCK 5 ITX - AI EDGE DEVICE**

### Thông tin cơ bản
- **Model:** Radxa Rock 5 ITX
- **RAM:** 8GB
- **Storage:** eMMC/SD Card (cần xác định)
- **OS:** Ubuntu/Rock OS (cần xác định)

### NPU - Neural Processing Unit
- **Model:** NPU integrated
- **Performance:** 6 TOPS
- **Architecture:** ARM-based
- **Power:** Low power consumption

### Vai trò trong dự án
- **Medium-Performance AI Inference**
- **Vehicle Analysis Models**
- **License Plate Recognition**
- **Vehicle Counting & Classification**

---

## 🚀 **JETSON NANO - AI EDGE DEVICE**

### Thông tin cơ bản
- **Model:** NVIDIA Jetson Nano
- **RAM:** 4GB LPDDR4
- **Storage:** 16GB eMMC
- **OS:** Ubuntu 18.04 LTS (JetPack)

### GPU - Maxwell Architecture
- **Model:** NVIDIA Maxwell GPU
- **CUDA Cores:** 128
- **Memory:** 4GB shared with system
- **Performance:** 472 GFLOPS

### Vai trò trong dự án
- **Lightweight AI Inference**
- **Prototype Development**
- **Testing & Validation**
- **Educational/Development Platform**

---

## 💻 **CORE i5 MACHINE - DEVELOPMENT/TESTING**

### Thông tin cơ bản
- **CPU:** Intel Core i5 (cần xác định model)
- **RAM:** 32GB
- **GPU:** 12GB (cần xác định model)
- **OS:** (cần xác định)

### Vai trò trong dự án
- **Secondary Development Machine**
- **Model Testing & Validation**
- **Performance Benchmarking**
- **Cross-platform Testing**

---

## 🎯 **AI MODELS TARGET**

### 👥 **Human Analysis Models**
1. **Face Recognition**
   - Face detection & identification
   - Age & gender estimation
   - Emotion recognition
   - Face quality assessment

2. **Person Detection & Counting**
   - Human detection in crowds
   - People counting
   - Crowd density estimation
   - Person tracking

3. **Behavior Analysis**
   - Action recognition
   - Pose estimation
   - Activity monitoring
   - Anomaly detection

### 🚗 **Vehicle Analysis Models**
1. **License Plate Recognition**
   - Plate detection & extraction
   - OCR for plate numbers
   - Multi-country support
   - Plate quality assessment

2. **Vehicle Classification**
   - Car, truck, bus, motorcycle detection
   - Brand & model recognition
   - Color classification
   - Vehicle type identification

3. **Vehicle Counting & Analytics**
   - Traffic flow counting
   - Vehicle speed estimation
   - Traffic pattern analysis
   - Congestion detection

---

## 🔄 **CI/CD PIPELINE ARCHITECTURE**

### Development Workflow
```
myai (Development) → GitLab/GitHub → CI/CD Pipeline → Docker Images → AI Devices
```

### Pipeline Stages
1. **Code Development** (myai + Cursor AI)
2. **Model Training** (myai + RTX 4060 Ti)
3. **Testing & Validation** (All devices)
4. **Docker Build** (myai)
5. **Image Push** (Registry)
6. **Deployment** (Target AI devices)

### Tools & Technologies
- **IDE:** Cursor AI
- **Version Control:** GitLab/GitHub
- **CI/CD:** GitHub Actions / GitLab CI
- **Containerization:** Docker
- **Registry:** Docker Hub / GitLab Registry
- **Deployment:** SSH / Docker Compose

---

## 📊 **DEVICE COMPARISON MATRIX**

| Device | CPU Cores | RAM | AI Performance | Power | Use Case |
|--------|-----------|-----|----------------|-------|----------|
| **myai** | 88 | 125GB | RTX 4060 Ti | High | Development/Training |
| **Raspberry Pi 5** | 4 | 8GB | 13 TOPS | Low | High-performance Edge |
| **Radxa Rock 5 ITX** | 6 | 8GB | 6 TOPS | Low | Medium Edge |
| **Jetson Nano** | 4 | 4GB | 472 GFLOPS | Very Low | Lightweight Edge |
| **Core i5** | 4-6 | 32GB | 12GB GPU | Medium | Testing/Validation |

---

## 🚀 **NEXT STEPS**

### Phase 1: Setup & Configuration
1. **Device Inventory Completion**
   - Gather missing specifications
   - Network configuration
   - OS versions & updates

2. **Development Environment**
   - Cursor AI setup on myai
   - GitLab/GitHub repository creation
   - CI/CD pipeline configuration

3. **AI Framework Setup**
   - TensorFlow/PyTorch installation
   - OpenCV & computer vision libraries
   - Model optimization tools

### Phase 2: Model Development
1. **Human Analysis Models**
   - Face recognition pipeline
   - Person detection & counting
   - Behavior analysis models

2. **Vehicle Analysis Models**
   - License plate recognition
   - Vehicle classification
   - Traffic analytics

### Phase 3: Deployment & Testing
1. **Docker Containerization**
2. **Multi-device deployment**
3. **Performance optimization**
4. **Real-world testing**

---

## 📝 **NOTES & UPDATES**

### Last Updated
- **Date:** 2025-08-30
- **Status:** Initial setup
- **Next Review:** After device inventory completion

### Pending Information
- [ ] Raspberry Pi 5 storage capacity
- [ ] Radxa Rock 5 ITX OS & storage details
- [ ] Core i5 machine specifications
- [ ] Network IP addresses for all devices
- [ ] SSH access configuration

---

## 🔗 **USEFUL COMMANDS**

### Device Management
```bash
# Check device status
ssh device_name "uname -a && free -h && df -h"

# Deploy Docker image
docker pull registry/image:tag
docker run -d --name ai_service registry/image:tag

# Monitor AI performance
nvidia-smi  # For GPU devices
htop        # For CPU monitoring
```

### Development Commands
```bash
# Train model on myai
python train_model.py --device cuda --epochs 100

# Build Docker image
docker build -t ai_model:latest .

# Push to registry
docker push registry/ai_model:latest
```

---

*Document này sẽ được cập nhật liên tục khi có thông tin mới về các thiết bị và tiến độ dự án.*
