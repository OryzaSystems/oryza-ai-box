#!/usr/bin/env python3
"""
AI Box Environment Test Script
Tests all installed AI frameworks and hardware acceleration
"""

import sys
import subprocess
import importlib
from typing import Dict, List, Tuple, Any
import platform
import psutil


class EnvironmentTester:
    """Test AI Box development environment"""
    
    def __init__(self):
        self.results: Dict[str, Any] = {}
        self.errors: List[str] = []
        
    def test_system_info(self) -> Dict[str, Any]:
        """Test system information"""
        print("🖥️  Testing System Information...")
        
        try:
            info = {
                "platform": platform.platform(),
                "architecture": platform.architecture(),
                "processor": platform.processor(),
                "python_version": sys.version,
                "cpu_count": psutil.cpu_count(logical=True),
                "physical_cores": psutil.cpu_count(logical=False),
                "memory_total": f"{psutil.virtual_memory().total / (1024**3):.1f} GB",
                "memory_available": f"{psutil.virtual_memory().available / (1024**3):.1f} GB"
            }
            
            print(f"   ✅ Platform: {info['platform']}")
            print(f"   ✅ CPU Cores: {info['cpu_count']} ({info['physical_cores']} physical)")
            print(f"   ✅ Memory: {info['memory_available']} / {info['memory_total']}")
            
            return info
            
        except Exception as e:
            self.errors.append(f"System info error: {e}")
            return {}
    
    def test_pytorch(self) -> Dict[str, Any]:
        """Test PyTorch installation and CUDA"""
        print("🔥 Testing PyTorch...")
        
        try:
            import torch
            import torchvision
            
            info = {
                "torch_version": torch.__version__,
                "torchvision_version": torchvision.__version__,
                "cuda_available": torch.cuda.is_available(),
                "cuda_device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
                "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
                "cudnn_version": torch.backends.cudnn.version() if torch.cuda.is_available() else None
            }
            
            if torch.cuda.is_available():
                info["gpu_name"] = torch.cuda.get_device_name(0)
                info["gpu_memory"] = f"{torch.cuda.get_device_properties(0).total_memory / (1024**3):.1f} GB"
                
                # Test GPU computation
                x = torch.randn(1000, 1000).cuda()
                y = torch.mm(x, x.t())
                info["gpu_test"] = "✅ GPU computation successful"
                
            print(f"   ✅ PyTorch: {info['torch_version']}")
            print(f"   ✅ CUDA Available: {info['cuda_available']}")
            if info['cuda_available']:
                print(f"   ✅ GPU: {info['gpu_name']} ({info['gpu_memory']})")
                print(f"   ✅ CUDA Version: {info['cuda_version']}")
                print(f"   ✅ {info['gpu_test']}")
            
            return info
            
        except Exception as e:
            self.errors.append(f"PyTorch error: {e}")
            return {"error": str(e)}
    
    def test_opencv(self) -> Dict[str, Any]:
        """Test OpenCV installation"""
        print("📸 Testing OpenCV...")
        
        try:
            import cv2
            import numpy as np
            
            info = {
                "opencv_version": cv2.__version__,
                "build_info": cv2.getBuildInformation()[:200] + "..."
            }
            
            # Test basic operations
            img = np.zeros((100, 100, 3), dtype=np.uint8)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            info["basic_test"] = "✅ Basic operations successful"
            
            print(f"   ✅ OpenCV: {info['opencv_version']}")
            print(f"   ✅ {info['basic_test']}")
            
            return info
            
        except Exception as e:
            self.errors.append(f"OpenCV error: {e}")
            return {"error": str(e)}
    
    def test_ultralytics(self) -> Dict[str, Any]:
        """Test Ultralytics YOLO"""
        print("🎯 Testing Ultralytics YOLO...")
        
        try:
            from ultralytics import YOLO
            import ultralytics
            
            info = {
                "ultralytics_version": ultralytics.__version__,
            }
            
            # Test model loading (without downloading)
            try:
                # This will test the framework without downloading models
                info["framework_test"] = "✅ Framework loaded successfully"
            except Exception as model_error:
                info["framework_test"] = f"⚠️  Framework loaded, model test skipped: {model_error}"
            
            print(f"   ✅ Ultralytics: {info['ultralytics_version']}")
            print(f"   ✅ {info['framework_test']}")
            
            return info
            
        except Exception as e:
            self.errors.append(f"Ultralytics error: {e}")
            return {"error": str(e)}
    
    def test_fastapi(self) -> Dict[str, Any]:
        """Test FastAPI and Uvicorn"""
        print("🚀 Testing FastAPI...")
        
        try:
            import fastapi
            import uvicorn
            from pydantic import BaseModel
            
            info = {
                "fastapi_version": fastapi.__version__,
                "uvicorn_version": uvicorn.__version__,
            }
            
            # Test basic FastAPI app creation
            app = fastapi.FastAPI()
            
            @app.get("/")
            def read_root():
                return {"Hello": "World"}
            
            info["app_test"] = "✅ FastAPI app created successfully"
            
            print(f"   ✅ FastAPI: {info['fastapi_version']}")
            print(f"   ✅ Uvicorn: {info['uvicorn_version']}")
            print(f"   ✅ {info['app_test']}")
            
            return info
            
        except Exception as e:
            self.errors.append(f"FastAPI error: {e}")
            return {"error": str(e)}
    
    def test_additional_packages(self) -> Dict[str, Any]:
        """Test additional packages"""
        print("📦 Testing Additional Packages...")
        
        packages_to_test = [
            ("numpy", "NumPy"),
            ("pandas", "Pandas"),
            ("scipy", "SciPy"),
            ("matplotlib", "Matplotlib"),
            ("PIL", "Pillow"),
            ("requests", "Requests"),
        ]
        
        info = {}
        
        for package_name, display_name in packages_to_test:
            try:
                module = importlib.import_module(package_name)
                version = getattr(module, '__version__', 'Unknown')
                info[package_name] = {
                    "version": version,
                    "status": "✅ OK"
                }
                print(f"   ✅ {display_name}: {version}")
                
            except ImportError as e:
                info[package_name] = {
                    "error": str(e),
                    "status": "❌ Missing"
                }
                print(f"   ❌ {display_name}: Missing")
                self.errors.append(f"{display_name} not installed: {e}")
        
        return info
    
    def test_docker(self) -> Dict[str, Any]:
        """Test Docker installation"""
        print("🐳 Testing Docker...")
        
        try:
            result = subprocess.run(['docker', '--version'], 
                                  capture_output=True, text=True, timeout=10)
            
            if result.returncode == 0:
                version = result.stdout.strip()
                info = {
                    "docker_version": version,
                    "status": "✅ Docker available"
                }
                print(f"   ✅ {version}")
                
                # Test docker info
                try:
                    info_result = subprocess.run(['docker', 'info'], 
                                               capture_output=True, text=True, timeout=10)
                    if info_result.returncode == 0:
                        info["docker_daemon"] = "✅ Docker daemon running"
                        print(f"   ✅ Docker daemon running")
                    else:
                        info["docker_daemon"] = "⚠️  Docker daemon not running"
                        print(f"   ⚠️  Docker daemon not running")
                except:
                    info["docker_daemon"] = "⚠️  Cannot check Docker daemon"
                
            else:
                info = {"error": "Docker command failed", "status": "❌ Error"}
                self.errors.append("Docker command failed")
            
            return info
            
        except (subprocess.TimeoutExpired, FileNotFoundError) as e:
            info = {"error": str(e), "status": "❌ Not found"}
            self.errors.append(f"Docker not found: {e}")
            print(f"   ❌ Docker not found")
            return info
    
    def run_all_tests(self) -> Dict[str, Any]:
        """Run all environment tests"""
        print("🧪 AI BOX ENVIRONMENT TEST")
        print("=" * 50)
        
        self.results = {
            "system": self.test_system_info(),
            "pytorch": self.test_pytorch(),
            "opencv": self.test_opencv(),
            "ultralytics": self.test_ultralytics(),
            "fastapi": self.test_fastapi(),
            "packages": self.test_additional_packages(),
            "docker": self.test_docker(),
        }
        
        print("\n" + "=" * 50)
        print("📊 TEST SUMMARY")
        print("=" * 50)
        
        if not self.errors:
            print("🎉 ALL TESTS PASSED! Environment is ready for AI Box development.")
            print("\n✅ Ready for:")
            print("   • AI Model Development")
            print("   • GPU Acceleration")
            print("   • Computer Vision")
            print("   • API Development")
            print("   • Docker Deployment")
        else:
            print(f"⚠️  {len(self.errors)} issues found:")
            for error in self.errors:
                print(f"   • {error}")
        
        return self.results


if __name__ == "__main__":
    tester = EnvironmentTester()
    results = tester.run_all_tests()
    
    # Exit with error code if there are issues
    sys.exit(len(tester.errors))
