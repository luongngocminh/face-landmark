#include "landmark_tracker.h"
#include <iostream>
#include <vector>
#include <memory>
#include <cmath>
#include <algorithm>

// Custom Matrix implementation to avoid external dependencies
namespace {
    // Simple Matrix class for Kalman filter implementation
    template<int Rows, int Cols>
    class Matrix {
    public:
        Matrix() {
            for (int i = 0; i < Rows * Cols; ++i) {
                data[i] = 0.0f;
            }
        }

        // Constructor with initialization list
        Matrix(std::initializer_list<float> values) {
            int i = 0;
            for (auto value : values) {
                if (i < Rows * Cols) {
                    data[i++] = value;
                }
            }
        }

        // Access element
        inline float& operator()(int row, int col) {
            return data[row * Cols + col];
        }

        inline const float& operator()(int row, int col) const {
            return data[row * Cols + col];
        }

        // Matrix addition
        Matrix<Rows, Cols> operator+(const Matrix<Rows, Cols>& other) const {
            Matrix<Rows, Cols> result;
            for (int i = 0; i < Rows * Cols; ++i) {
                result.data[i] = data[i] + other.data[i];
            }
            return result;
        }

        // Matrix subtraction
        Matrix<Rows, Cols> operator-(const Matrix<Rows, Cols>& other) const {
            Matrix<Rows, Cols> result;
            for (int i = 0; i < Rows * Cols; ++i) {
                result.data[i] = data[i] - other.data[i];
            }
            return result;
        }

        // Matrix multiplication
        template<int OtherCols>
        Matrix<Rows, OtherCols> operator*(const Matrix<Cols, OtherCols>& other) const {
            Matrix<Rows, OtherCols> result;
            for (int i = 0; i < Rows; ++i) {
                for (int j = 0; j < OtherCols; ++j) {
                    float sum = 0.0f;
                    for (int k = 0; k < Cols; ++k) {
                        sum += (*this)(i, k) * other(k, j);
                    }
                    result(i, j) = sum;
                }
            }
            return result;
        }

        // Transpose
        Matrix<Cols, Rows> transpose() const {
            Matrix<Cols, Rows> result;
            for (int i = 0; i < Rows; ++i) {
                for (int j = 0; j < Cols; ++j) {
                    result(j, i) = (*this)(i, j);
                }
            }
            return result;
        }

        // Create identity matrix
        static Matrix<Rows, Cols> Identity() {
            static_assert(Rows == Cols, "Identity matrix must be square");
            Matrix<Rows, Cols> result;
            for (int i = 0; i < Rows; ++i) {
                result(i, i) = 1.0f;
            }
            return result;
        }

        // Matrix inverse (specialized for 2x2 matrices used in Kalman filter)
        Matrix<Rows, Cols> inverse() const {
            static_assert(Rows == Cols, "Only square matrices can be inverted");
            
            Matrix<Rows, Cols> result;
            
            if constexpr (Rows == 2 && Cols == 2) {
                // 2x2 matrix inversion
                float det = (*this)(0, 0) * (*this)(1, 1) - (*this)(0, 1) * (*this)(1, 0);
                if (std::abs(det) < 1e-6f) {
                    // Handle nearly singular matrix
                    const float epsilon = 1e-6f;
                    det = (det >= 0) ? std::max(det, epsilon) : std::min(det, -epsilon);
                }
                
                float invDet = 1.0f / det;
                result(0, 0) = (*this)(1, 1) * invDet;
                result(0, 1) = -(*this)(0, 1) * invDet;
                result(1, 0) = -(*this)(1, 0) * invDet;
                result(1, 1) = (*this)(0, 0) * invDet;
            } else {
                // For this application, we only need 2x2 inverse matrices
                return Matrix<Rows, Cols>::Identity();
            }
            
            return result;
        }

    private:
        float data[Rows * Cols];
    };

    // Type aliases for commonly used matrices
    using Vector2f = Matrix<2, 1>;
    using Vector4f = Matrix<4, 1>;
    using Matrix2f = Matrix<2, 2>;
    using Matrix4f = Matrix<4, 4>;
    using Matrix4x2f = Matrix<4, 2>;
    using Matrix2x4f = Matrix<2, 4>;
}

// Define Kalman tracker for a single landmark point
class KalmanTracker {
public:
    KalmanTracker() {
        // Initialize Kalman filter
        // State vector: [x, y, vx, vy]
        // Measurement vector: [x, y]
        
        // State transition matrix
        F = {1, 0, 1, 0,
             0, 1, 0, 1,
             0, 0, 1, 0,
             0, 0, 0, 1};
             
        // Measurement matrix
        H = {1, 0, 0, 0,
             0, 1, 0, 0};
             
        // Process noise covariance
        Q = {0.01, 0, 0, 0,
             0, 0.01, 0, 0,
             0, 0, 0.05, 0,
             0, 0, 0, 0.05};
             
        // Measurement noise covariance
        R = {0.5, 0,
             0, 0.5};
             
        // Initial state covariance
        P = {1, 0, 0, 0,
             0, 1, 0, 0,
             0, 0, 1, 0,
             0, 0, 0, 1};
             
        // Initial state
        x = {0, 0, 0, 0};
        
        // Identity matrix
        I = Matrix4f::Identity();
        
        initialized = false;
    }
    
    // Initialize with first observation
    void init(float x_pos, float y_pos) {
        x(0, 0) = x_pos;
        x(1, 0) = y_pos;
        x(2, 0) = 0;
        x(3, 0) = 0;
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
        
        Vector2f z;
        z(0, 0) = x_pos;
        z(1, 0) = y_pos;
        
        // Compute innovation
        Vector2f y = z - H * x;
        
        // Innovation covariance
        Matrix2f S = H * P * H.transpose() + R;
        
        // Compute Kalman gain
        Matrix4x2f K = P * H.transpose() * S.inverse();
        
        // Update state with measurement
        x = x + K * y;
        
        // Update error covariance
        P = (I - K * H) * P;
        
        // Apply additional temporal smoothing if needed
        if (smoothingFactor < 1.0f) {
            x(0, 0) = prevX(0, 0) * (1.0f - smoothingFactor) + x(0, 0) * smoothingFactor;
            x(1, 0) = prevX(1, 0) * (1.0f - smoothingFactor) + x(1, 0) * smoothingFactor;
        }
        
        // Store current state for next frame smoothing
        prevX = x;
    }
    
    // Get current position
    std::pair<float, float> getPosition() const {
        return std::make_pair(x(0, 0), x(1, 0));
    }
    
    // Check if initialized
    bool isInitialized() const {
        return initialized;
    }
    
    // Reset tracker
    void reset() {
        initialized = false;
        x = {0, 0, 0, 0};
        P = {1, 0, 0, 0,
             0, 1, 0, 0,
             0, 0, 1, 0,
             0, 0, 0, 1};
    }

private:
    bool initialized;
    Vector4f x;         // State vector [x, y, vx, vy]
    Vector4f prevX;     // Previous state for smoothing
    Matrix4f P;         // Error covariance
    Matrix4f F;         // State transition matrix
    Matrix2x4f H;       // Measurement matrix
    Matrix4f Q;         // Process noise covariance
    Matrix2f R;         // Measurement noise covariance
    Matrix4f I;         // Identity matrix
};

// Implementation of LandmarkTracker
class LandmarkTracker::Impl {
public:
    Impl(int numLandmarks) 
        : numLandmarks(numLandmarks), 
          stabilizationEnabled(true),
          temporalSmoothingFactor(0.8f),
          keepOriginalIndices(true) {
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
    bool keepOriginalIndices;  // Controls whether to maintain original indices
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
