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

// Hungarian algorithm for point association with full implementation
class HungarianAlgorithm {
public:
    std::vector<int> solve(const std::vector<std::vector<float>>& costMatrix) {
        int n = costMatrix.size();
        int m = n > 0 ? costMatrix[0].size() : 0;
        std::vector<int> assignment(n, -1);
        
        if (n == 0 || m == 0) {
            return assignment;
        }
        
        // Make a square cost matrix by padding with zeros if necessary
        int size = std::max(n, m);
        std::vector<std::vector<float>> squareCost(size, std::vector<float>(size, 0));
        
        // Copy the original cost matrix to the square matrix
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < m; j++) {
                squareCost[i][j] = costMatrix[i][j];
            }
        }
        
        // Initialize necessary vectors
        std::vector<int> rowAssignment(size, -1);
        std::vector<int> colAssignment(size, -1);
        std::vector<float> rowDual(size, 0);
        std::vector<float> colDual(size, 0);
        
        // Step 1: Reduce the matrix by subtracting the minimum value from each row
        for (int i = 0; i < size; i++) {
            float minVal = std::numeric_limits<float>::max();
            for (int j = 0; j < size; j++) {
                minVal = std::min(minVal, squareCost[i][j]);
            }
            for (int j = 0; j < size; j++) {
                squareCost[i][j] -= minVal;
            }
            rowDual[i] = minVal;
        }
        
        // Step 2: Reduce the matrix by subtracting the minimum value from each column
        for (int j = 0; j < size; j++) {
            float minVal = std::numeric_limits<float>::max();
            for (int i = 0; i < size; i++) {
                minVal = std::min(minVal, squareCost[i][j]);
            }
            for (int i = 0; i < size; i++) {
                squareCost[i][j] -= minVal;
            }
            colDual[j] = minVal;
        }
        
        // Step 3: Find a minimum set of zeros that covers all rows and columns
        std::vector<bool> rowCovered(size, false);
        std::vector<bool> colCovered(size, false);
        
        // Find an initial assignment
        for (int i = 0; i < size; i++) {
            for (int j = 0; j < size; j++) {
                if (squareCost[i][j] == 0 && rowAssignment[i] == -1 && colAssignment[j] == -1) {
                    rowAssignment[i] = j;
                    colAssignment[j] = i;
                }
            }
        }
        
        // Count the number of assignments
        int numAssigned = 0;
        for (int i = 0; i < size; i++) {
            if (rowAssignment[i] != -1) {
                numAssigned++;
            }
        }
        
        // While not all rows are assigned
        while (numAssigned < size) {
            // Find an unassigned row
            int unassignedRow = -1;
            for (int i = 0; i < size; i++) {
                if (rowAssignment[i] == -1) {
                    unassignedRow = i;
                    break;
                }
            }
            
            // Reset coverages
            std::fill(rowCovered.begin(), rowCovered.end(), false);
            std::fill(colCovered.begin(), colCovered.end(), false);
            
            // Find augmenting path
            std::vector<int> path;
            path.push_back(unassignedRow);
            
            bool foundPath = false;
            int currentRow = unassignedRow;
            
            while (!foundPath) {
                // Find uncovered zero in current row
                int minCol = -1;
                float minVal = std::numeric_limits<float>::max();
                
                for (int j = 0; j < size; j++) {
                    if (!colCovered[j]) {
                        if (squareCost[currentRow][j] < minVal) {
                            minVal = squareCost[currentRow][j];
                            minCol = j;
                        }
                    }
                }
                
                // If no zeros found, update the dual problem
                if (minVal > 0) {
                    // Find smallest uncovered value
                    float theta = minVal;
                    
                    // Subtract from uncovered rows
                    for (int i = 0; i < size; i++) {
                        if (!rowCovered[i]) {
                            for (int j = 0; j < size; j++) {
                                if (!colCovered[j]) {
                                    squareCost[i][j] -= theta;
                                }
                            }
                        }
                    }
                    
                    // Add to covered columns
                    for (int j = 0; j < size; j++) {
                        if (colCovered[j]) {
                            for (int i = 0; i < size; i++) {
                                squareCost[i][j] += theta;
                            }
                        }
                    }
                    
                    continue;
                }
                
                // Found a zero at (currentRow, minCol)
                colCovered[minCol] = true;
                
                // Check if the column is assigned
                int assignedRow = colAssignment[minCol];
                if (assignedRow == -1) {
                    // Augmenting path found
                    int currentCol = minCol;
                    
                    // Update assignments along the path
                    for (int i = 0; i < path.size(); i++) {
                        int nextRow = path[i];
                        int nextCol = i < path.size() - 1 ? rowAssignment[path[i + 1]] : currentCol;
                        
                        rowAssignment[nextRow] = nextCol;
                        colAssignment[nextCol] = nextRow;
                    }
                    
                    numAssigned++;
                    foundPath = true;
                } else {
                    // Continue finding the path
                    currentRow = assignedRow;
                    rowCovered[currentRow] = true;
                    path.push_back(currentRow);
                }
            }
        }
        
        // Extract assignments for the original matrix dimensions
        for (int i = 0; i < n; i++) {
            if (rowAssignment[i] != -1 && rowAssignment[i] < m) {
                assignment[i] = rowAssignment[i];
            }
        }
        
        return assignment;
    }
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
        hungarian = std::make_unique<HungarianAlgorithm>();
        
        // Initialize landmark groups
        landmarkGroups.resize(numLandmarks, 0);
        defineAnatomicalGroups();
    }
    
    // Define anatomical landmark groups to prevent mismatching between different facial features
    void defineAnatomicalGroups() {
        // Nhóm các điểm landmark theo vùng giải phẫu để tránh hoán đổi
        // Mặc định, tất cả các điểm thuộc nhóm 0
        
        // Phân nhóm cho 106 điểm landmark của khuôn mặt
        // Các nhóm: 1-17: đường viền khuôn mặt
        for (int i = 0; i < 17; i++) {
            landmarkGroups[i] = 1; // Đường viền khuôn mặt
        }
        
        // 18-27: lông mày trái
        for (int i = 17; i < 27; i++) {
            landmarkGroups[i] = 2; // Lông mày trái
        }
        
        // 28-37: lông mày phải
        for (int i = 27; i < 37; i++) {
            landmarkGroups[i] = 3; // Lông mày phải
        }
        
        // 38-49: mũi
        for (int i = 37; i < 49; i++) {
            landmarkGroups[i] = 4; // Mũi
        }
        
        // 50-61: mắt trái
        for (int i = 49; i < 61; i++) {
            landmarkGroups[i] = 5; // Mắt trái
        }
        
        // 62-73: mắt phải
        for (int i = 61; i < 73; i++) {
            landmarkGroups[i] = 6; // Mắt phải
        }
        
        // 74-95: môi ngoài và môi trong
        for (int i = 73; i < 95; i++) {
            landmarkGroups[i] = 7; // Môi
        }
        
        // 96-106: các điểm trong miệng
        for (int i = 95; i < 106; i++) {
            landmarkGroups[i] = 8; // Trong miệng
        }
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
        
        // Method 1: Direct assignment based on original indices with high stability
        if (keepOriginalIndices) {
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
        
        // Method 2: Use Hungarian with constraints and improvements
        // Compute cost matrix for Hungarian algorithm with stricter distance penalties
        std::vector<std::vector<float>> costMatrix(numLandmarks, std::vector<float>(numLandmarks));
        
        // Define anatomical constraints for facial landmarks
        // For example, landmarks from left eye should only match with left eye landmarks
        defineAnatomicalGroups();
        
        const float MAX_ALLOWED_DISTANCE = 30.0f; //  Reduce allowed distance
        const float PENALTY_MULTIPLIER = 10.0f;    // Increase penalty for moving between groups
        
        for (int i = 0; i < numLandmarks; i++) {
            auto predicted = trackers[i].getPosition();
            float pred_x = predicted.first;
            float pred_y = predicted.second;
            int sourceGroup = landmarkGroups[i];
            
            for (int j = 0; j < numLandmarks; j++) {
                float obs_x = landmarks[j * 2];
                float obs_y = landmarks[j * 2 + 1];
                int targetGroup = landmarkGroups[j];
                
                // Euclidean distance as base cost
                float dx = pred_x - obs_x;
                float dy = pred_y - obs_y;
                float distance = std::sqrt(dx*dx + dy*dy);
                
                // Add strong penalty for matching across different anatomical groups
                if (sourceGroup != targetGroup) {
                    costMatrix[i][j] = distance + PENALTY_MULTIPLIER * MAX_ALLOWED_DISTANCE;
                } else {
                    costMatrix[i][j] = distance;
                }
                
                // Add strong penalty for large distances
                if (distance > MAX_ALLOWED_DISTANCE) {
                    costMatrix[i][j] += PENALTY_MULTIPLIER * (distance - MAX_ALLOWED_DISTANCE);
                }
                
                // Add smaller penalty for matching with different indices
                if (i != j) {
                    // Smaller penalty for nearby landmarks, larger for distant ones
                    float indexDistance = std::abs(i - j);
                    costMatrix[i][j] += indexDistance * 0.5f;
                }
            }
        }
        
        // Solve assignment problem with improved Hungarian algorithm
        std::vector<int> assignments = hungarian->solve(costMatrix);
        
        // Second pass: update Kalman filters with assigned measurements
        for (int i = 0; i < numLandmarks; i++) {
            int assignedIdx = assignments[i];
            
            if (assignedIdx >= 0 && assignedIdx < numLandmarks) {
                float x = landmarks[assignedIdx * 2];
                float y = landmarks[assignedIdx * 2 + 1];
                
                // Only update if the assigned point is not too far from prediction
                auto predicted = trackers[i].getPosition();
                float dx = predicted.first - x;
                float dy = predicted.second - y;
                float distance = std::sqrt(dx*dx + dy*dy);
                
                // Stricter maximum allowed distance
                if (distance <= MAX_ALLOWED_DISTANCE) {
                    // Verify that the assigned landmark is from the same anatomical group
                    if (landmarkGroups[i] == landmarkGroups[assignedIdx]) {
                        trackers[i].update(x, y, temporalSmoothingFactor);
                    } else {
                        // Use original position if cross-group assignment occurred
                        trackers[i].update(landmarks[i * 2], landmarks[i * 2 + 1], temporalSmoothingFactor);
                    }
                } else {
                    // Use original index if distance is too large
                    // Apply stronger smoothing when rejecting assignments
                    trackers[i].update(landmarks[i * 2], landmarks[i * 2 + 1], temporalSmoothingFactor * 0.8f);
                }
            } else {
                // No assignment, use original index
                trackers[i].update(landmarks[i * 2], landmarks[i * 2 + 1], temporalSmoothingFactor);
            }
            
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
    std::unique_ptr<HungarianAlgorithm> hungarian;
    bool firstFrame = true;
    bool stabilizationEnabled;
    float temporalSmoothingFactor;
    bool keepOriginalIndices;  // Kiểm soát việc duy trì chỉ số gốc
    std::vector<int> landmarkGroups; // Nhóm giải phẫu của các landmark
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
