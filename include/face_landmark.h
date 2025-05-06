#ifndef FACE_LANDMARK_H
#define FACE_LANDMARK_H

#include <vector>
#include <string>
#include <memory>

// Define attributes properly for different contexts
#ifdef __EMSCRIPTEN__
#include <emscripten/emscripten.h>
#define EXPORT EMSCRIPTEN_KEEPALIVE
#else
#define EXPORT
#endif

// Add SIMD support if enabled
#ifdef WITH_SIMD
#define SIMD_ENABLED 1
#else
#define SIMD_ENABLED 0
#endif

// Define the ROI structure
struct ROI {
    float x1, y1, x2, y2; // Coordinates of the bounding box
    float score;         // Confidence score
    int label;           // Label of the detected object
};

// Proper use of the EXPORT attribute for a class
class FaceLandmarkDetector {
public:
    EXPORT FaceLandmarkDetector();
    EXPORT ~FaceLandmarkDetector();
    
    // Prevent copying and moving
    FaceLandmarkDetector(const FaceLandmarkDetector&) = delete;
    FaceLandmarkDetector& operator=(const FaceLandmarkDetector&) = delete;
    FaceLandmarkDetector(FaceLandmarkDetector&&) = delete;
    FaceLandmarkDetector& operator=(FaceLandmarkDetector&&) = delete;

    EXPORT bool initialize();
    EXPORT bool loadModel(const std::string& modelPath);
    
    // Process raw image data (RGBA format)
    EXPORT std::vector<float> detectLandmarks(const std::vector<unsigned char>& imageData, int width, int height);
        
    // Thread-safe methods
    EXPORT void setNumThreads(int numThreads);
    EXPORT int getNumThreads() const;

    // Added method to check if SIMD is enabled
    EXPORT bool isSIMDEnabled();
    
    // Landmark stabilization control
    EXPORT void setStabilizationEnabled(bool enabled);
    EXPORT bool isStabilizationEnabled() const;
    EXPORT void setTemporalSmoothing(float factor);
    EXPORT float getTemporalSmoothing() const;

private:
    // PIMPL idiom - forward declaration of implementation class
    class Impl;
    
    // Use unique_ptr to manage the implementation object
    std::unique_ptr<Impl> pImpl;
};

#endif // FACE_LANDMARK_H
