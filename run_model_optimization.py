#!/usr/bin/env python3
"""
Complete Model Optimization Workflow

This script orchestrates the complete model optimization process:
1. Model Quantization (INT8, FP16, Dynamic)
2. Platform-Specific Optimization (TensorRT, Hailo, RKNN, OpenVINO)
3. Performance Analysis & Comparison
4. Comprehensive Reporting
"""

import os
import sys
import time
import json
import logging
import argparse
from datetime import datetime
from typing import Dict, List, Any, Optional
from pathlib import Path

# Add tools directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'tools'))

# Import optimization tools
try:
    from model_quantizer import ModelQuantizer, QuantizationConfig
    from platform_optimizer import PlatformOptimizer, PlatformConfig
    from optimization_analyzer import OptimizationAnalyzer
    from performance_monitor import PerformanceMonitor
except ImportError as e:
    print(f"❌ Error importing optimization tools: {e}")
    sys.exit(1)

# Import AI models
try:
    from ai_models import (
        FaceDetector, FaceRecognizer, PersonDetector, BehaviorAnalyzer,
        VehicleDetector, LicensePlateOCR, VehicleClassifier, TrafficAnalyzer
    )
    from ai_models.common.model_config import ModelConfig
    AI_MODELS_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ Warning: Could not import AI models: {e}")
    AI_MODELS_AVAILABLE = False

class ModelOptimizationWorkflow:
    """Complete model optimization workflow orchestrator."""
    
    def __init__(self, output_dir: str = "optimization_results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        self.logger = self._setup_logging()
        
        # Initialize optimization tools
        self.quantizer = ModelQuantizer(str(self.output_dir / "quantized"))
        self.platform_optimizer = PlatformOptimizer(str(self.output_dir / "platform_optimized"))
        self.analyzer = OptimizationAnalyzer(str(self.output_dir / "analysis"))
        self.performance_monitor = PerformanceMonitor(str(self.output_dir / "performance.json"))
        
        # Workflow configuration
        self.config = {
            'enable_quantization': True,
            'enable_platform_optimization': True,
            'enable_analysis': True,
            'target_platforms': ['tensorrt', 'hailo', 'rknn', 'openvino'],
            'quantization_methods': ['int8', 'fp16', 'dynamic'],
            'performance_target_fps': 30,
            'accuracy_threshold': 0.90
        }
        
        # Results storage
        self.optimization_results = {}
        
    def _setup_logging(self) -> logging.Logger:
        """Setup logging configuration."""
        log_file = self.output_dir / "optimization_workflow.log"
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        return logging.getLogger(__name__)
        
    def create_demo_models(self) -> Dict[str, Any]:
        """Create demo models for optimization testing."""
        import torch
        import torch.nn as nn
        
        class DemoFaceDetector(nn.Module):
            def __init__(self):
                super().__init__()
                self.backbone = nn.Sequential(
                    nn.Conv2d(3, 32, 3, padding=1),
                    nn.ReLU(),
                    nn.Conv2d(32, 64, 3, padding=1),
                    nn.ReLU(),
                    nn.AdaptiveAvgPool2d((7, 7))
                )
                self.classifier = nn.Linear(64 * 7 * 7, 2)  # face/no-face
                
            def forward(self, x):
                x = self.backbone(x)
                x = x.view(x.size(0), -1)
                x = self.classifier(x)
                return x
        
        class DemoVehicleDetector(nn.Module):
            def __init__(self):
                super().__init__()
                self.backbone = nn.Sequential(
                    nn.Conv2d(3, 64, 3, padding=1),
                    nn.ReLU(),
                    nn.Conv2d(64, 128, 3, padding=1),
                    nn.ReLU(),
                    nn.AdaptiveAvgPool2d((7, 7))
                )
                self.classifier = nn.Linear(128 * 7 * 7, 5)  # 5 vehicle types
                
            def forward(self, x):
                x = self.backbone(x)
                x = x.view(x.size(0), -1)
                x = self.classifier(x)
                return x
        
        models = {
            'FaceDetector': DemoFaceDetector(),
            'VehicleDetector': DemoVehicleDetector()
        }
        
        return models
        
    def optimize_model_quantization(self, model, model_name: str) -> Dict[str, Any]:
        """Optimize model using quantization methods."""
        self.logger.info(f"🔧 Starting quantization optimization for {model_name}...")
        
        quantization_results = {}
        
        # Measure original performance
        self.logger.info(f"📊 Measuring baseline performance for {model_name}...")
        original_perf = self.analyzer.measure_model_performance(
            model, model_name, "original", "myai"
        )
        
        # Apply quantization methods
        for method in self.config['quantization_methods']:
            try:
                self.logger.info(f"🔧 Applying {method.upper()} quantization to {model_name}...")
                
                if method == 'int8':
                    result = self.quantizer.quantize_model_int8(model, model_name)
                elif method == 'fp16':
                    result = self.quantizer.quantize_model_fp16(model, model_name)
                elif method == 'dynamic':
                    result = self.quantizer.quantize_model_dynamic(model, model_name)
                
                quantization_results[method] = result
                
                if result.success:
                    self.logger.info(f"✅ {method.upper()} quantization completed: {result.speedup:.1f}x speedup")
                else:
                    self.logger.error(f"❌ {method.upper()} quantization failed: {result.error_message}")
                    
            except Exception as e:
                self.logger.error(f"❌ {method.upper()} quantization error: {e}")
                
        return {
            'original_performance': original_perf,
            'quantization_results': quantization_results
        }
        
    def optimize_model_platforms(self, model, model_name: str) -> Dict[str, Any]:
        """Optimize model for different platforms."""
        self.logger.info(f"🚀 Starting platform optimization for {model_name}...")
        
        platform_results = {}
        
        # Apply platform optimizations
        for platform in self.config['target_platforms']:
            try:
                self.logger.info(f"🔧 Optimizing {model_name} for {platform}...")
                
                config = PlatformConfig(
                    platform_name=platform,
                    device_type="myai",
                    precision="fp16"
                )
                
                result = self.platform_optimizer.optimize_for_platform(
                    model, model_name, platform, config
                )
                
                platform_results[platform] = result
                
                if result.success:
                    self.logger.info(f"✅ {platform} optimization completed: {result.speedup:.1f}x speedup")
                else:
                    self.logger.error(f"❌ {platform} optimization failed: {result.error_message}")
                    
            except Exception as e:
                self.logger.error(f"❌ {platform} optimization error: {e}")
                
        return platform_results
        
    def analyze_optimization_results(self, model_name: str) -> Dict[str, Any]:
        """Analyze optimization results for a model."""
        self.logger.info(f"📊 Analyzing optimization results for {model_name}...")
        
        try:
            analysis = self.analyzer.analyze_optimization_impact(model_name)
            
            if "error" not in analysis:
                self.logger.info(f"✅ Analysis completed for {model_name}")
            else:
                self.logger.warning(f"⚠️ Analysis warning: {analysis['error']}")
                
            return analysis
            
        except Exception as e:
            self.logger.error(f"❌ Analysis error for {model_name}: {e}")
            return {"error": str(e)}
            
    def optimize_single_model(self, model, model_name: str) -> Dict[str, Any]:
        """Complete optimization workflow for a single model."""
        self.logger.info(f"🚀 Starting complete optimization for {model_name}...")
        
        start_time = time.time()
        
        results = {
            'model_name': model_name,
            'start_time': datetime.now().isoformat(),
            'quantization': {},
            'platform_optimization': {},
            'analysis': {},
            'success': False
        }
        
        try:
            # Step 1: Quantization optimization
            if self.config['enable_quantization']:
                self.logger.info(f"📋 Step 1: Quantization optimization for {model_name}")
                results['quantization'] = self.optimize_model_quantization(model, model_name)
            
            # Step 2: Platform optimization
            if self.config['enable_platform_optimization']:
                self.logger.info(f"📋 Step 2: Platform optimization for {model_name}")
                results['platform_optimization'] = self.optimize_model_platforms(model, model_name)
            
            # Step 3: Analysis
            if self.config['enable_analysis']:
                self.logger.info(f"📋 Step 3: Optimization analysis for {model_name}")
                results['analysis'] = self.analyze_optimization_results(model_name)
            
            results['success'] = True
            results['duration'] = time.time() - start_time
            
            self.logger.info(f"✅ Complete optimization finished for {model_name} in {results['duration']:.1f}s")
            
        except Exception as e:
            results['error'] = str(e)
            results['duration'] = time.time() - start_time
            self.logger.error(f"❌ Optimization failed for {model_name}: {e}")
            
        return results
        
    def optimize_all_models(self, models: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize all provided models."""
        self.logger.info(f"🚀 Starting optimization for {len(models)} models...")
        
        all_results = {
            'workflow_start_time': datetime.now().isoformat(),
            'total_models': len(models),
            'model_results': {},
            'summary': {}
        }
        
        # Optimize each model
        for model_name, model in models.items():
            self.logger.info(f"🔄 Processing model {model_name}...")
            
            model_results = self.optimize_single_model(model, model_name)
            all_results['model_results'][model_name] = model_results
            
            # Store in class results
            self.optimization_results[model_name] = model_results
            
        # Generate summary
        all_results['summary'] = self._generate_workflow_summary()
        all_results['workflow_end_time'] = datetime.now().isoformat()
        
        return all_results
        
    def _generate_workflow_summary(self) -> Dict[str, Any]:
        """Generate workflow summary."""
        successful_models = [r for r in self.optimization_results.values() if r.get('success', False)]
        failed_models = [r for r in self.optimization_results.values() if not r.get('success', False)]
        
        summary = {
            'total_models': len(self.optimization_results),
            'successful_optimizations': len(successful_models),
            'failed_optimizations': len(failed_models),
            'success_rate': len(successful_models) / len(self.optimization_results) * 100 if self.optimization_results else 0,
            'avg_duration': sum(r.get('duration', 0) for r in successful_models) / len(successful_models) if successful_models else 0
        }
        
        return summary
        
    def generate_comprehensive_report(self) -> Dict[str, Any]:
        """Generate comprehensive optimization report."""
        self.logger.info("📄 Generating comprehensive optimization report...")
        
        # Get individual tool reports
        quantization_report = self.quantizer.generate_quantization_report()
        platform_report = self.platform_optimizer.generate_optimization_report()
        analysis_report = self.analyzer.generate_optimization_report()
        
        comprehensive_report = {
            'workflow_summary': self._generate_workflow_summary(),
            'quantization_report': quantization_report,
            'platform_optimization_report': platform_report,
            'analysis_report': analysis_report,
            'model_results': self.optimization_results,
            'configuration': self.config,
            'timestamp': datetime.now().isoformat()
        }
        
        return comprehensive_report
        
    def save_comprehensive_report(self, filename: str = "comprehensive_optimization_report.json"):
        """Save comprehensive report to file."""
        report = self.generate_comprehensive_report()
        
        report_path = self.output_dir / filename
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
            
        self.logger.info(f"📄 Comprehensive report saved to {report_path}")
        
    def print_workflow_summary(self):
        """Print workflow summary to console."""
        summary = self._generate_workflow_summary()
        
        print("\n" + "="*80)
        print("🔧 MODEL OPTIMIZATION WORKFLOW SUMMARY")
        print("="*80)
        
        print(f"📊 OVERVIEW:")
        print(f"   Total Models: {summary['total_models']}")
        print(f"   Successful Optimizations: {summary['successful_optimizations']}")
        print(f"   Failed Optimizations: {summary['failed_optimizations']}")
        print(f"   Success Rate: {summary['success_rate']:.1f}%")
        print(f"   Average Duration: {summary['avg_duration']:.1f}s per model")
        
        # Print individual tool summaries
        print(f"\n🔧 QUANTIZATION SUMMARY:")
        self.quantizer.print_quantization_summary()
        
        print(f"\n🚀 PLATFORM OPTIMIZATION SUMMARY:")
        self.platform_optimizer.print_optimization_summary()
        
        print(f"\n📊 ANALYSIS SUMMARY:")
        self.analyzer.print_analysis_summary()
        
        print("="*80)
        
    def run_complete_workflow(self, models: Optional[Dict[str, Any]] = None):
        """Run complete optimization workflow."""
        self.logger.info("🚀 Starting complete model optimization workflow...")
        
        # Use demo models if none provided
        if models is None:
            self.logger.info("📋 Creating demo models for optimization...")
            models = self.create_demo_models()
        
        # Run optimization
        results = self.optimize_all_models(models)
        
        # Generate reports
        self.save_comprehensive_report()
        
        # Print summary
        self.print_workflow_summary()
        
        self.logger.info("✅ Complete optimization workflow finished successfully!")
        
        return results

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Model Optimization Workflow")
    parser.add_argument('--output', type=str, default='optimization_results', help='Output directory')
    parser.add_argument('--no-quantization', action='store_true', help='Disable quantization')
    parser.add_argument('--no-platform', action='store_true', help='Disable platform optimization')
    parser.add_argument('--no-analysis', action='store_true', help='Disable analysis')
    
    args = parser.parse_args()
    
    print("🔧 Model Optimization Workflow")
    print("="*50)
    print("This workflow optimizes AI models using:")
    print("- Quantization (INT8, FP16, Dynamic)")
    print("- Platform optimization (TensorRT, Hailo, RKNN, OpenVINO)")
    print("- Performance analysis and comparison")
    print("="*50)
    
    # Create workflow
    workflow = ModelOptimizationWorkflow(args.output)
    
    # Update configuration
    workflow.config.update({
        'enable_quantization': not args.no_quantization,
        'enable_platform_optimization': not args.no_platform,
        'enable_analysis': not args.no_analysis
    })
    
    # Run workflow
    workflow.run_complete_workflow()

if __name__ == "__main__":
    main()
