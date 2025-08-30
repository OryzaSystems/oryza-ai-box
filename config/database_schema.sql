-- ==========================================
-- AI BOX - Database Schema Design
-- PostgreSQL Database Schema for AI Box System
-- ==========================================

-- Enable UUID extension
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pgcrypto";

-- ==========================================
-- SYSTEM TABLES
-- ==========================================

-- System Configuration
CREATE TABLE system_config (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    config_key VARCHAR(100) NOT NULL UNIQUE,
    config_value TEXT,
    description TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Device Management
CREATE TABLE devices (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    device_name VARCHAR(100) NOT NULL,
    device_type VARCHAR(50) NOT NULL, -- 'raspberry-pi-5', 'radxa-rock-5', 'jetson-nano', 'core-i5'
    ip_address INET,
    mac_address MACADDR,
    platform_info JSONB, -- CPU, RAM, GPU/NPU specs
    status VARCHAR(20) DEFAULT 'offline', -- 'online', 'offline', 'maintenance'
    last_heartbeat TIMESTAMP WITH TIME ZONE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- ==========================================
-- USER MANAGEMENT
-- ==========================================

-- Users
CREATE TABLE users (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    username VARCHAR(50) NOT NULL UNIQUE,
    email VARCHAR(100) NOT NULL UNIQUE,
    password_hash VARCHAR(255) NOT NULL,
    full_name VARCHAR(100),
    role VARCHAR(20) DEFAULT 'user', -- 'admin', 'operator', 'viewer', 'user'
    is_active BOOLEAN DEFAULT true,
    last_login TIMESTAMP WITH TIME ZONE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- User Sessions
CREATE TABLE user_sessions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    session_token VARCHAR(255) NOT NULL UNIQUE,
    device_info JSONB,
    ip_address INET,
    expires_at TIMESTAMP WITH TIME ZONE NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- ==========================================
-- AI MODEL MANAGEMENT
-- ==========================================

-- AI Models
CREATE TABLE ai_models (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    model_name VARCHAR(100) NOT NULL,
    model_type VARCHAR(50) NOT NULL, -- 'face_recognition', 'person_detection', 'vehicle_classification', etc.
    model_version VARCHAR(20) NOT NULL,
    model_path TEXT NOT NULL,
    model_config JSONB, -- Model-specific configuration
    supported_platforms TEXT[], -- ['raspberry-pi-5', 'radxa-rock-5', 'jetson-nano']
    performance_metrics JSONB, -- FPS, accuracy, memory usage per platform
    is_active BOOLEAN DEFAULT true,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    UNIQUE(model_name, model_version)
);

-- Model Deployments
CREATE TABLE model_deployments (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    model_id UUID NOT NULL REFERENCES ai_models(id) ON DELETE CASCADE,
    device_id UUID NOT NULL REFERENCES devices(id) ON DELETE CASCADE,
    deployment_config JSONB, -- Device-specific config
    status VARCHAR(20) DEFAULT 'pending', -- 'pending', 'deployed', 'failed', 'stopped'
    deployed_at TIMESTAMP WITH TIME ZONE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    UNIQUE(model_id, device_id)
);

-- ==========================================
-- HUMAN ANALYSIS TABLES
-- ==========================================

-- Face Recognition
CREATE TABLE faces (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    person_id UUID, -- Reference to persons table (optional)
    face_encoding BYTEA NOT NULL, -- Face embedding vector
    face_image_path TEXT,
    confidence_score FLOAT,
    bounding_box JSONB, -- {x, y, width, height}
    detected_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    device_id UUID REFERENCES devices(id),
    camera_id VARCHAR(50),
    metadata JSONB -- age, gender, emotion, etc.
);

-- Person Detection
CREATE TABLE person_detections (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    detection_id VARCHAR(100) NOT NULL, -- Unique detection ID
    person_count INTEGER NOT NULL,
    bounding_boxes JSONB NOT NULL, -- Array of bounding boxes
    image_path TEXT,
    confidence_scores FLOAT[],
    detected_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    device_id UUID REFERENCES devices(id),
    camera_id VARCHAR(50),
    metadata JSONB -- crowd density, activities, etc.
);

-- Behavior Analysis
CREATE TABLE behavior_analysis (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    detection_id VARCHAR(100) NOT NULL,
    behavior_type VARCHAR(50) NOT NULL, -- 'walking', 'running', 'standing', 'sitting', 'fighting', etc.
    confidence_score FLOAT NOT NULL,
    bounding_box JSONB,
    duration_seconds INTEGER,
    analyzed_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    device_id UUID REFERENCES devices(id),
    camera_id VARCHAR(50),
    metadata JSONB
);

-- ==========================================
-- VEHICLE ANALYSIS TABLES
-- ==========================================

-- License Plate Recognition
CREATE TABLE license_plates (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    plate_number VARCHAR(20) NOT NULL,
    country_code VARCHAR(3),
    plate_type VARCHAR(20), -- 'car', 'motorcycle', 'truck', 'bus'
    confidence_score FLOAT NOT NULL,
    bounding_box JSONB,
    plate_image_path TEXT,
    vehicle_image_path TEXT,
    detected_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    device_id UUID REFERENCES devices(id),
    camera_id VARCHAR(50),
    metadata JSONB
);

-- Vehicle Classification
CREATE TABLE vehicle_detections (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    detection_id VARCHAR(100) NOT NULL,
    vehicle_type VARCHAR(50) NOT NULL, -- 'car', 'truck', 'bus', 'motorcycle', 'bicycle'
    brand VARCHAR(50),
    model VARCHAR(50),
    color VARCHAR(30),
    confidence_score FLOAT NOT NULL,
    bounding_box JSONB,
    image_path TEXT,
    detected_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    device_id UUID REFERENCES devices(id),
    camera_id VARCHAR(50),
    metadata JSONB -- speed, direction, etc.
);

-- Traffic Analytics
CREATE TABLE traffic_analytics (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    analytics_type VARCHAR(50) NOT NULL, -- 'flow', 'congestion', 'violation', 'speed'
    location_id VARCHAR(100),
    vehicle_count INTEGER,
    average_speed FLOAT,
    congestion_level VARCHAR(20), -- 'low', 'medium', 'high', 'critical'
    violations JSONB, -- Array of violation types
    time_period TIMESTAMP WITH TIME ZONE NOT NULL,
    duration_minutes INTEGER DEFAULT 5,
    device_id UUID REFERENCES devices(id),
    camera_id VARCHAR(50),
    raw_data JSONB
);

-- ==========================================
-- SYSTEM MONITORING
-- ==========================================

-- System Metrics
CREATE TABLE system_metrics (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    device_id UUID NOT NULL REFERENCES devices(id) ON DELETE CASCADE,
    metric_type VARCHAR(50) NOT NULL, -- 'cpu', 'memory', 'gpu', 'disk', 'network'
    metric_value FLOAT NOT NULL,
    metric_unit VARCHAR(20), -- '%', 'MB', 'GB', 'fps', etc.
    recorded_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    metadata JSONB
);

-- API Logs
CREATE TABLE api_logs (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    request_id VARCHAR(100),
    method VARCHAR(10) NOT NULL,
    endpoint TEXT NOT NULL,
    status_code INTEGER,
    response_time_ms INTEGER,
    user_id UUID REFERENCES users(id),
    device_id UUID REFERENCES devices(id),
    ip_address INET,
    user_agent TEXT,
    request_body JSONB,
    response_body JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Error Logs
CREATE TABLE error_logs (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    error_type VARCHAR(50) NOT NULL,
    error_message TEXT NOT NULL,
    stack_trace TEXT,
    device_id UUID REFERENCES devices(id),
    service_name VARCHAR(50), -- 'api-gateway', 'model-server', 'data-manager'
    severity VARCHAR(20) DEFAULT 'error', -- 'debug', 'info', 'warning', 'error', 'critical'
    context JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- ==========================================
-- INDEXES FOR PERFORMANCE
-- ==========================================

-- Users
CREATE INDEX idx_users_username ON users(username);
CREATE INDEX idx_users_email ON users(email);
CREATE INDEX idx_users_role ON users(role);

-- Devices
CREATE INDEX idx_devices_type ON devices(device_type);
CREATE INDEX idx_devices_status ON devices(status);
CREATE INDEX idx_devices_ip ON devices(ip_address);

-- AI Models
CREATE INDEX idx_ai_models_type ON ai_models(model_type);
CREATE INDEX idx_ai_models_active ON ai_models(is_active);

-- Detection Tables (Time-based queries)
CREATE INDEX idx_faces_detected_at ON faces(detected_at);
CREATE INDEX idx_faces_device_camera ON faces(device_id, camera_id);
CREATE INDEX idx_person_detections_detected_at ON person_detections(detected_at);
CREATE INDEX idx_person_detections_device_camera ON person_detections(device_id, camera_id);
CREATE INDEX idx_license_plates_detected_at ON license_plates(detected_at);
CREATE INDEX idx_license_plates_number ON license_plates(plate_number);
CREATE INDEX idx_vehicle_detections_detected_at ON vehicle_detections(detected_at);
CREATE INDEX idx_vehicle_detections_type ON vehicle_detections(vehicle_type);

-- Analytics
CREATE INDEX idx_traffic_analytics_time_period ON traffic_analytics(time_period);
CREATE INDEX idx_traffic_analytics_location ON traffic_analytics(location_id);
CREATE INDEX idx_traffic_analytics_type ON traffic_analytics(analytics_type);

-- Monitoring
CREATE INDEX idx_system_metrics_device_time ON system_metrics(device_id, recorded_at);
CREATE INDEX idx_system_metrics_type ON system_metrics(metric_type);
CREATE INDEX idx_api_logs_created_at ON api_logs(created_at);
CREATE INDEX idx_error_logs_created_at ON error_logs(created_at);
CREATE INDEX idx_error_logs_severity ON error_logs(severity);

-- ==========================================
-- SAMPLE DATA
-- ==========================================

-- Insert default system configuration
INSERT INTO system_config (config_key, config_value, description) VALUES
('system_name', 'Oryza AI Box', 'System name'),
('version', '1.0.0', 'System version'),
('max_devices', '10', 'Maximum number of devices'),
('retention_days', '30', 'Data retention period in days'),
('enable_face_recognition', 'true', 'Enable face recognition feature'),
('enable_vehicle_analysis', 'true', 'Enable vehicle analysis feature'),
('api_rate_limit', '1000', 'API rate limit per hour'),
('log_level', 'INFO', 'System log level');

-- Insert default admin user (password: admin123)
INSERT INTO users (username, email, password_hash, full_name, role) VALUES
('admin', 'admin@oryza.vn', crypt('admin123', gen_salt('bf')), 'System Administrator', 'admin'),
('operator', 'operator@oryza.vn', crypt('operator123', gen_salt('bf')), 'System Operator', 'operator'),
('viewer', 'viewer@oryza.vn', crypt('viewer123', gen_salt('bf')), 'System Viewer', 'viewer');

-- Insert sample AI models
INSERT INTO ai_models (model_name, model_type, model_version, model_path, supported_platforms, performance_metrics) VALUES
('YOLOv8-Face', 'face_detection', '1.0.0', '/models/yolov8_face.pt', 
 ARRAY['raspberry-pi-5', 'radxa-rock-5', 'jetson-nano', 'core-i5'],
 '{"raspberry-pi-5": {"fps": 15, "accuracy": 0.92}, "radxa-rock-5": {"fps": 12, "accuracy": 0.90}, "jetson-nano": {"fps": 8, "accuracy": 0.88}, "core-i5": {"fps": 25, "accuracy": 0.95}}'::jsonb),

('FaceNet', 'face_recognition', '1.0.0', '/models/facenet.h5',
 ARRAY['raspberry-pi-5', 'core-i5'],
 '{"raspberry-pi-5": {"fps": 10, "accuracy": 0.95}, "core-i5": {"fps": 30, "accuracy": 0.98}}'::jsonb),

('YOLOv8-Person', 'person_detection', '1.0.0', '/models/yolov8_person.pt',
 ARRAY['raspberry-pi-5', 'radxa-rock-5', 'jetson-nano', 'core-i5'],
 '{"raspberry-pi-5": {"fps": 20, "accuracy": 0.88}, "radxa-rock-5": {"fps": 18, "accuracy": 0.86}, "jetson-nano": {"fps": 12, "accuracy": 0.84}, "core-i5": {"fps": 35, "accuracy": 0.92}}'::jsonb),

('YOLOv8-Vehicle', 'vehicle_detection', '1.0.0', '/models/yolov8_vehicle.pt',
 ARRAY['raspberry-pi-5', 'radxa-rock-5', 'jetson-nano', 'core-i5'],
 '{"raspberry-pi-5": {"fps": 18, "accuracy": 0.90}, "radxa-rock-5": {"fps": 16, "accuracy": 0.88}, "jetson-nano": {"fps": 10, "accuracy": 0.86}, "core-i5": {"fps": 32, "accuracy": 0.94}}'::jsonb),

('PaddleOCR', 'license_plate_ocr', '2.7.0', '/models/paddleocr',
 ARRAY['raspberry-pi-5', 'radxa-rock-5', 'jetson-nano', 'core-i5'],
 '{"raspberry-pi-5": {"fps": 5, "accuracy": 0.95}, "radxa-rock-5": {"fps": 8, "accuracy": 0.93}, "jetson-nano": {"fps": 3, "accuracy": 0.91}, "core-i5": {"fps": 15, "accuracy": 0.98}}'::jsonb);

-- ==========================================
-- FUNCTIONS AND TRIGGERS
-- ==========================================

-- Function to update updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Apply updated_at trigger to relevant tables
CREATE TRIGGER update_system_config_updated_at BEFORE UPDATE ON system_config FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
CREATE TRIGGER update_devices_updated_at BEFORE UPDATE ON devices FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
CREATE TRIGGER update_users_updated_at BEFORE UPDATE ON users FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
CREATE TRIGGER update_ai_models_updated_at BEFORE UPDATE ON ai_models FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
CREATE TRIGGER update_model_deployments_updated_at BEFORE UPDATE ON model_deployments FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- Function to clean old data based on retention policy
CREATE OR REPLACE FUNCTION cleanup_old_data()
RETURNS void AS $$
DECLARE
    retention_days INTEGER;
BEGIN
    -- Get retention period from config
    SELECT config_value::INTEGER INTO retention_days 
    FROM system_config 
    WHERE config_key = 'retention_days';
    
    IF retention_days IS NULL THEN
        retention_days := 30; -- Default 30 days
    END IF;
    
    -- Clean old detection data
    DELETE FROM faces WHERE detected_at < NOW() - INTERVAL '1 day' * retention_days;
    DELETE FROM person_detections WHERE detected_at < NOW() - INTERVAL '1 day' * retention_days;
    DELETE FROM behavior_analysis WHERE analyzed_at < NOW() - INTERVAL '1 day' * retention_days;
    DELETE FROM license_plates WHERE detected_at < NOW() - INTERVAL '1 day' * retention_days;
    DELETE FROM vehicle_detections WHERE detected_at < NOW() - INTERVAL '1 day' * retention_days;
    DELETE FROM traffic_analytics WHERE time_period < NOW() - INTERVAL '1 day' * retention_days;
    
    -- Clean old logs
    DELETE FROM system_metrics WHERE recorded_at < NOW() - INTERVAL '1 day' * retention_days;
    DELETE FROM api_logs WHERE created_at < NOW() - INTERVAL '1 day' * retention_days;
    DELETE FROM error_logs WHERE created_at < NOW() - INTERVAL '1 day' * retention_days;
    
    -- Clean expired sessions
    DELETE FROM user_sessions WHERE expires_at < NOW();
END;
$$ LANGUAGE plpgsql;

-- ==========================================
-- VIEWS FOR COMMON QUERIES
-- ==========================================

-- Active devices with latest metrics
CREATE VIEW active_devices_with_metrics AS
SELECT 
    d.*,
    sm.cpu_usage,
    sm.memory_usage,
    sm.gpu_usage,
    sm.last_metric_time
FROM devices d
LEFT JOIN (
    SELECT 
        device_id,
        MAX(CASE WHEN metric_type = 'cpu' THEN metric_value END) as cpu_usage,
        MAX(CASE WHEN metric_type = 'memory' THEN metric_value END) as memory_usage,
        MAX(CASE WHEN metric_type = 'gpu' THEN metric_value END) as gpu_usage,
        MAX(recorded_at) as last_metric_time
    FROM system_metrics 
    WHERE recorded_at > NOW() - INTERVAL '5 minutes'
    GROUP BY device_id
) sm ON d.id = sm.device_id
WHERE d.status = 'online';

-- Daily analytics summary
CREATE VIEW daily_analytics_summary AS
SELECT 
    DATE(time_period) as analytics_date,
    analytics_type,
    location_id,
    COUNT(*) as record_count,
    AVG(vehicle_count) as avg_vehicle_count,
    AVG(average_speed) as avg_speed,
    COUNT(CASE WHEN congestion_level = 'high' OR congestion_level = 'critical' THEN 1 END) as high_congestion_periods
FROM traffic_analytics
WHERE time_period >= CURRENT_DATE - INTERVAL '30 days'
GROUP BY DATE(time_period), analytics_type, location_id
ORDER BY analytics_date DESC, analytics_type, location_id;
