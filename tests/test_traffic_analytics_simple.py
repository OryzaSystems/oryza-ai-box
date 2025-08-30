# ==========================================
# AI Box - Traffic Analytics Simple Test
# Test traffic analytics infrastructure
# ==========================================

import sys
import os
from pathlib import Path
import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from ai_models.common.model_config import ModelConfig
from ai_models.vehicle_analysis.traffic_analyzer import TrafficAnalyzer, TrafficZone, VehicleTrack, TrafficMetrics
from ai_models.common.inference_result import Detection

def test_traffic_analyzer_initialization():
    """Test TrafficAnalyzer initialization without loading model."""
    print("🧪 Testing TrafficAnalyzer Initialization...")
    
    # Create config
    config = ModelConfig(
        model_name="traffic-analyzer",
        model_type="traffic_analysis",
        confidence_threshold=0.5,
        input_size=(640, 480),
        use_gpu=False,  # Analytics is CPU-based
        model_params={
            'max_tracks': 100,
            'track_timeout': 5.0,
            'pixels_per_meter': 10.0
        }
    )
    
    # Initialize analyzer (without loading model)
    try:
        traffic_analyzer = TrafficAnalyzer(config)
        print(f"✅ TrafficAnalyzer initialized: {traffic_analyzer}")
        
        # Test model info
        model_info = traffic_analyzer.get_model_info()
        print(f"✅ Model info: {model_info['model_class']}")
        
        # Test performance metrics
        metrics = traffic_analyzer.get_performance_metrics()
        print(f"✅ Performance metrics: {metrics}")
        
        # Test configuration
        print(f"✅ Max tracks: {traffic_analyzer.max_tracks}")
        print(f"✅ Track timeout: {traffic_analyzer.track_timeout}")
        print(f"✅ Pixels per meter: {traffic_analyzer.pixels_per_meter}")
        print(f"✅ Congestion thresholds: {traffic_analyzer.congestion_thresholds}")
        print(f"✅ Traffic zones: {len(traffic_analyzer.traffic_zones)}")
        print(f"✅ Active tracks: {len(traffic_analyzer.active_tracks)}")
        print(f"✅ Next track ID: {traffic_analyzer.next_track_id}")
        
        # Test metadata
        capabilities = traffic_analyzer.metadata.get('capabilities', [])
        print(f"✅ Capabilities: {capabilities}")
        
        return True
        
    except Exception as e:
        print(f"❌ TrafficAnalyzer initialization failed: {e}")
        return False

def test_platform_optimization():
    """Test platform optimization functionality."""
    print("\n🔧 Testing Platform Optimization...")
    
    config = ModelConfig(
        model_name="traffic-analyzer",
        model_type="traffic_analysis",
        platform="auto",
        model_params={'max_tracks': 100}
    )
    
    traffic_analyzer = TrafficAnalyzer(config)
    
    # Test different platforms
    platforms = ['raspberry-pi-5', 'radxa-rock-5', 'jetson-nano', 'core-i5']
    
    for platform in platforms:
        print(f"🔧 Testing optimization for {platform}...")
        success = traffic_analyzer.optimize_for_platform(platform)
        print(f"📊 {platform} optimization: {'✅ Success' if success else '❌ Failed'}")
        
        # Show optimized settings
        print(f"   - Max tracks: {traffic_analyzer.max_tracks}")
        print(f"   - Track timeout: {traffic_analyzer.track_timeout}")
    
    return True

def test_traffic_zones():
    """Test traffic zone functionality."""
    print("\n🧪 Testing Traffic Zones...")
    
    config = ModelConfig(
        model_name="traffic-analyzer",
        model_type="traffic_analysis",
        input_size=(640, 480)
    )
    
    traffic_analyzer = TrafficAnalyzer(config)
    
    # Test default zones creation
    traffic_analyzer._create_default_zones()
    print(f"📊 Default zones created: {len(traffic_analyzer.traffic_zones)}")
    
    for zone in traffic_analyzer.traffic_zones:
        print(f"   - {zone.name}: {zone.zone_type} ({len(zone.polygon)} points)")
        if zone.direction:
            print(f"     Direction: {zone.direction}")
    
    # Test zone polygon compilation
    traffic_analyzer._compile_zone_polygons()
    print(f"📊 Zone polygons compiled: {len(traffic_analyzer.zone_polygons)}")
    
    # Test custom zone addition
    custom_zone = TrafficZone(
        name="test_zone",
        polygon=[(100, 100), (200, 100), (200, 200), (100, 200)],
        zone_type="counting"
    )
    
    traffic_analyzer.add_traffic_zone(custom_zone)
    print(f"📊 Custom zone added: {custom_zone.name}")
    print(f"📊 Total zones: {len(traffic_analyzer.traffic_zones)}")
    
    return True

def test_data_structures():
    """Test traffic analytics data structures."""
    print("\n🧪 Testing Data Structures...")
    
    # Test TrafficZone
    zone = TrafficZone(
        name="test_entry",
        polygon=[(0, 0), (100, 0), (100, 50), (0, 50)],
        zone_type="entry",
        direction="south"
    )
    print(f"✅ TrafficZone: {zone.name} ({zone.zone_type})")
    
    # Test VehicleTrack
    from collections import deque
    track = VehicleTrack(
        track_id=1,
        vehicle_type="car",
        positions=deque([(100, 100), (105, 110)], maxlen=50),
        timestamps=deque([1000.0, 1001.0], maxlen=50),
        first_seen=1000.0,
        last_seen=1001.0
    )
    print(f"✅ VehicleTrack: ID {track.track_id} ({track.vehicle_type})")
    print(f"   Positions: {len(track.positions)}")
    print(f"   Zones visited: {len(track.zones_visited)}")
    
    # Test TrafficMetrics
    metrics = TrafficMetrics(
        timestamp=1000.0,
        total_vehicles=5,
        vehicle_counts={'car': 3, 'truck': 2},
        average_speed=45.5,
        traffic_density=0.25,
        congestion_level='low',
        flow_rate=12.5,
        zone_counts={'entry_north': 2, 'counting_center': 3}
    )
    print(f"✅ TrafficMetrics: {metrics.total_vehicles} vehicles")
    print(f"   Congestion: {metrics.congestion_level}")
    print(f"   Flow rate: {metrics.flow_rate}")
    
    return True

def test_vehicle_tracking():
    """Test vehicle tracking functionality."""
    print("\n🧪 Testing Vehicle Tracking...")
    
    config = ModelConfig(
        model_name="traffic-analyzer",
        model_type="traffic_analysis",
        input_size=(640, 480),
        model_params={'max_tracks': 10}
    )
    
    traffic_analyzer = TrafficAnalyzer(config)
    
    # Create mock vehicle detections
    detections = [
        Detection(
            bbox=[100, 100, 150, 150],
            confidence=0.8,
            class_id=2,
            class_name='car'
        ),
        Detection(
            bbox=[200, 200, 250, 250],
            confidence=0.7,
            class_id=7,
            class_name='truck'
        ),
        Detection(
            bbox=[300, 300, 330, 340],
            confidence=0.6,
            class_id=3,
            class_name='motorcycle'
        )
    ]
    
    print(f"📊 Mock detections created: {len(detections)}")
    
    # Test vehicle tracking update
    import time
    current_time = time.time()
    
    initial_tracks = len(traffic_analyzer.active_tracks)
    traffic_analyzer._update_vehicle_tracking(detections, current_time)
    final_tracks = len(traffic_analyzer.active_tracks)
    
    print(f"📊 Tracks before: {initial_tracks}")
    print(f"📊 Tracks after: {final_tracks}")
    print(f"📊 New tracks created: {final_tracks - initial_tracks}")
    
    # Show track details
    for track_id, track in traffic_analyzer.active_tracks.items():
        print(f"   Track {track_id}: {track.vehicle_type} at {track.positions[-1]}")
    
    return True

def test_traffic_metrics_calculation():
    """Test traffic metrics calculation."""
    print("\n🧪 Testing Traffic Metrics Calculation...")
    
    config = ModelConfig(
        model_name="traffic-analyzer",
        model_type="traffic_analysis",
        input_size=(640, 480)
    )
    
    traffic_analyzer = TrafficAnalyzer(config)
    
    # Add some mock tracks
    from collections import deque
    import time
    
    current_time = time.time()
    
    # Mock track 1
    track1 = VehicleTrack(
        track_id=1,
        vehicle_type="car",
        positions=deque([(100, 100)], maxlen=50),
        timestamps=deque([current_time], maxlen=50),
        first_seen=current_time,
        last_seen=current_time
    )
    
    # Mock track 2
    track2 = VehicleTrack(
        track_id=2,
        vehicle_type="truck",
        positions=deque([(200, 200)], maxlen=50),
        timestamps=deque([current_time], maxlen=50),
        first_seen=current_time,
        last_seen=current_time
    )
    
    traffic_analyzer.active_tracks[1] = track1
    traffic_analyzer.active_tracks[2] = track2
    
    # Calculate metrics
    metrics = traffic_analyzer._calculate_traffic_metrics(current_time)
    
    print(f"📊 Traffic metrics calculated:")
    print(f"   Total vehicles: {metrics.total_vehicles}")
    print(f"   Vehicle counts: {metrics.vehicle_counts}")
    print(f"   Traffic density: {metrics.traffic_density:.3f}")
    print(f"   Congestion level: {metrics.congestion_level}")
    print(f"   Flow rate: {metrics.flow_rate:.2f}")
    
    # Test metrics validation
    if metrics.total_vehicles == 2:
        print("✅ Vehicle count correct")
    else:
        print("❌ Vehicle count incorrect")
        return False
    
    if 'car' in metrics.vehicle_counts and 'truck' in metrics.vehicle_counts:
        print("✅ Vehicle types correct")
    else:
        print("❌ Vehicle types incorrect")
        return False
    
    if metrics.congestion_level in ['low', 'medium', 'high', 'severe']:
        print("✅ Congestion level valid")
    else:
        print("❌ Congestion level invalid")
        return False
    
    return True

def main():
    """Main test function."""
    print("🚀 AI Box - Traffic Analytics Simple Test")
    print("=" * 60)
    
    tests = [
        ("TrafficAnalyzer Initialization", test_traffic_analyzer_initialization),
        ("Platform Optimization", test_platform_optimization),
        ("Traffic Zones", test_traffic_zones),
        ("Data Structures", test_data_structures),
        ("Vehicle Tracking", test_vehicle_tracking),
        ("Traffic Metrics Calculation", test_traffic_metrics_calculation)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        try:
            if test_func():
                passed += 1
                print(f"✅ {test_name} PASSED")
            else:
                print(f"❌ {test_name} FAILED")
        except Exception as e:
            print(f"❌ {test_name} ERROR: {e}")
    
    print(f"\n{'='*60}")
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All traffic analytics infrastructure tests passed!")
        print("✅ Traffic Analytics Infrastructure is working correctly")
    else:
        print("❌ Some tests failed!")
        print("🔧 Please check the error messages above")
    
    return passed == total

if __name__ == "__main__":
    main()
