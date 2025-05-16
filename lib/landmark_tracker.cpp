#include "landmark_tracker.h"
#include <iostream>
#include <vector>
#include <memory>
#include <Eigen/Dense>
#include <unordered_map>
#include <cmath>
#include <algorithm>

// Define Kalman tracker for a single landmark point
class KalmanTracker {
public:
    KalmanTracker() {
        // Initialize Kalman filter
        // State vector: [x, y, vx, vy]
        // Measurement vector: [x, y]
        
        // State transition matrix
        F << 1, 0, 1, 0,
             0, 1, 0, 1,
             0, 0, 1, 0,
             0, 0, 0, 1;
             
        // Measurement matrix
        H << 1, 0, 0, 0,
             0, 1, 0, 0;
             
        // Process noise covariance
        Q << 0.01, 0, 0, 0,
             0, 0.01, 0, 0,
             0, 0, 0.05, 0,
             0, 0, 0, 0.05;
             
        // Measurement noise covariance
        R << 0.5, 0,
             0, 0.5;
             
        // Initial state covariance
        P << 1, 0, 0, 0,
             0, 1, 0, 0,
             0, 0, 1, 0,
             0, 0, 0, 1;
             
        // Initial state
        x << 0, 0, 0, 0;
        
        // Identity matrix
        I = Eigen::Matrix4f::Identity();
        
        initialized = false;
    }
    
    // Initialize with first observation
    void init(float x_pos, float y_pos) {
        x << x_pos, y_pos, 0, 0;
        initialized = true;
    }
    
    // Predict next state
    void predict() {
        if (!initialized) return;
        
        // Predict state
        x = F * x;
        
        // Update error covariance
        P = F * P * F.transpose() + Q;
    }
    
    // Update with new measurement
    void update(float x_pos, float y_pos, float smoothingFactor = 0.8f) {
        if (!initialized) {
            init(x_pos, y_pos);
            return;
        }
        
        Eigen::Vector2f z;
        z << x_pos, y_pos;
        
        // Compute Kalman gain
        Eigen::Matrix<float, 4, 2> K = P * H.transpose() * (H * P * H.transpose() + R).inverse();
        
        // Update state with measurement
        x = x + K * (z - H * x);
        
        // Update error covariance
        P = (I - K * H) * P;
        
        // Apply additional temporal smoothing if needed
        if (smoothingFactor < 1.0f) {
            x(0) = prevX(0) * (1.0f - smoothingFactor) + x(0) * smoothingFactor;
            x(1) = prevX(1) * (1.0f - smoothingFactor) + x(1) * smoothingFactor;
        }
        
        // Store current state for next frame smoothing
        prevX = x;
    }
    
    // Get current position
    std::pair<float, float> getPosition() const {
        return std::make_pair(x(0), x(1));
    }
    
    // Check if initialized
    bool isInitialized() const {
        return initialized;
    }
    
    // Reset tracker
    void reset() {
        initialized = false;
        x << 0, 0, 0, 0;
        P << 1, 0, 0, 0,
             0, 1, 0, 0,
             0, 0, 1, 0,
             0, 0, 0, 1;
    }

private:
    bool initialized;
    Eigen::Vector4f x;         // State vector [x, y, vx, vy]
    Eigen::Vector4f prevX;     // Previous state for smoothing
    Eigen::Matrix4f P;         // Error covariance
    Eigen::Matrix4f F;         // State transition matrix
    Eigen::Matrix<float, 2, 4> H; // Measurement matrix
    Eigen::Matrix4f Q;         // Process noise covariance
    Eigen::Matrix2f R;         // Measurement noise covariance
    Eigen::Matrix4f I;         // Identity matrix
};

// Implementation of LandmarkTracker
class LandmarkTracker::Impl {
public:
    Impl(int numLandmarks) 
        : numLandmarks(numLandmarks), 
          stabilizationEnabled(true),
          temporalSmoothingFactor(0.8f),
          keepOriginalIndices(true) { // Mặc định giữ chỉ số ban đầu để tránh hoán đổi vị trí
        // Initialize trackers
        trackers.resize(numLandmarks);
    }
    
    std::vector<float> update(const std::vector<float>& landmarks) {
        // Check if we have the correct number of landmarks
        if (landmarks.size() != numLandmarks * 2) {
            std::cerr << "Warning: Expected " << numLandmarks * 2 
                      << " values, got " << landmarks.size() << std::endl;
            return landmarks; // Return original landmarks if count doesn't match
        }
        
        // If stabilization is disabled, just return the input landmarks
        if (!stabilizationEnabled) {
            return landmarks;
        }
        
        std::vector<float> stabilizedLandmarks(numLandmarks * 2);
        
        // First pass: predict all kalman filters
        for (int i = 0; i < numLandmarks; i++) {
            trackers[i].predict();
        }
        
        // If this is the first frame, initialize all trackers
        if (firstFrame) {
            for (int i = 0; i < numLandmarks; i++) {
                float x = landmarks[i * 2];
                float y = landmarks[i * 2 + 1];
                trackers[i].init(x, y);
            }
            firstFrame = false;
            
            // Return landmarks as-is for first frame
            return landmarks;
        }
        
        // Direct assignment based on original indices with high stability
        for (int i = 0; i < numLandmarks; i++) {
            float x = landmarks[i * 2];
            float y = landmarks[i * 2 + 1];
            
            // Use Kalman filter only for smoothing, no reassignment
            trackers[i].update(x, y, temporalSmoothingFactor);
            
            // Get stabilized position
            auto position = trackers[i].getPosition();
            stabilizedLandmarks[i * 2] = position.first;
            stabilizedLandmarks[i * 2 + 1] = position.second;
        }
        
        return stabilizedLandmarks;
    }
    
    void reset() {
        for (auto& tracker : trackers) {
            tracker.reset();
        }
        firstFrame = true;
    }
    
    void setStabilizationEnabled(bool enabled) {
        stabilizationEnabled = enabled;
        if (!enabled) {
            reset();
        }
    }
    
    bool isStabilizationEnabled() const {
        return stabilizationEnabled;
    }
    
    void setTemporalSmoothing(float factor) {
        // Clamp factor between 0 and 1
        temporalSmoothingFactor = std::max(0.0f, std::min(1.0f, factor));
    }
    
    float getTemporalSmoothing() const {
        return temporalSmoothingFactor;
    }
    
private:
    int numLandmarks;
    std::vector<KalmanTracker> trackers;
    bool firstFrame = true;
    bool stabilizationEnabled;
    float temporalSmoothingFactor;
    bool keepOriginalIndices;  // Kiểm soát việc duy trì chỉ số gốc
};

// Public LandmarkTracker implementation
LandmarkTracker::LandmarkTracker(int numLandmarks)
    : pImpl(std::make_unique<Impl>(numLandmarks)) {
}

LandmarkTracker::~LandmarkTracker() = default;

std::vector<float> LandmarkTracker::update(const std::vector<float>& landmarks) {
    return pImpl->update(landmarks);
}

void LandmarkTracker::reset() {
    pImpl->reset();
}

void LandmarkTracker::setStabilizationEnabled(bool enabled) {
    pImpl->setStabilizationEnabled(enabled);
}

bool LandmarkTracker::isStabilizationEnabled() const {
    return pImpl->isStabilizationEnabled();
}

void LandmarkTracker::setTemporalSmoothing(float factor) {
    pImpl->setTemporalSmoothing(factor);
}

float LandmarkTracker::getTemporalSmoothing() const {
    return pImpl->getTemporalSmoothing();
}
