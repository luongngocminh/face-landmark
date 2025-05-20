#ifndef LANDMARK_TRACKER_H
#define LANDMARK_TRACKER_H

#include <vector>
#include <memory>

// Forward declarations
class KalmanTracker;

/**
 * @brief LandmarkTracker class that stabilizes facial landmarks across frames
 * 
 * This class uses Kalman filtering to track and stabilize 106 facial landmarks
 * in real-time. It maintains the state of each landmark point and uses Hungarian
 * algorithm for point matching between frames.
 */
class LandmarkTracker {
public:
    /**
     * @brief Constructor for LandmarkTracker
     * @param numLandmarks Number of landmarks to track (default: 106)
     */
    LandmarkTracker(int numLandmarks = 106);
    
    /**
     * @brief Destructor
     */
    ~LandmarkTracker();
    
    /**
     * @brief Update the tracker with new landmarks
     * @param landmarks Vector of x,y coordinates (x0,y0,x1,y1,...,x105,y105)
     * @return Vector of stabilized landmarks
     */
    std::vector<float> update(const std::vector<float>& landmarks);
    
    /**
     * @brief Reset the tracker state
     */
    void reset();
    
    /**
     * @brief Enable or disable stabilization
     * @param enabled Whether stabilization should be enabled
     */
    void setStabilizationEnabled(bool enabled);
    
    /**
     * @brief Check if stabilization is enabled
     * @return True if stabilization is enabled, false otherwise
     */
    bool isStabilizationEnabled() const;
    
    /**
     * @brief Set temporal smoothing factor (0.0 - 1.0)
     * 
     * Lower values result in more smoothing but more latency
     * Higher values result in less smoothing but less latency
     * 
     * @param factor Smoothing factor (0.0 - 1.0)
     */
    void setTemporalSmoothing(float factor);
    
    /**
     * @brief Get current temporal smoothing factor
     * @return Current smoothing factor
     */
    float getTemporalSmoothing() const;
    
private:
    // Implementation details
    class Impl;
    std::unique_ptr<Impl> pImpl;
};

#endif // LANDMARK_TRACKER_H
