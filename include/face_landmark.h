#ifndef FACE_LANDMARK_H
#define FACE_LANDMARK_H

#include <vector>
#include <string>
#include <mutex>
#include <thread>

// Include NCNN headers
#include "net.h"
#include "layer.h"
#include "mat.h"

// Always include Emscripten since we're targeting web only
#include <emscripten/emscripten.h>

// Define attributes properly for different contexts
#ifdef __EMSCRIPTEN__
#define EXPORT EMSCRIPTEN_KEEPALIVE
#else
#define EXPORT
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

    EXPORT bool initialize();
    EXPORT bool loadModel(const std::string& modelPath);
    
    // Process raw image data (RGBA format)
    EXPORT std::vector<float> detectLandmarks(const std::vector<unsigned char>& imageData, int width, int height);
    
    // Process NCNN Mat directly
    EXPORT std::vector<float> detectLandmarksFromMat(const ncnn::Mat& image);
    
    // Thread-safe methods
    EXPORT void setNumThreads(int numThreads);
    EXPORT int getNumThreads() const;

private:
    // NCNN-specific processing
    ncnn::Mat preprocessImage(const ncnn::Mat& image);
    std::vector<ROI> extractROI(const ncnn::Mat& image);
    std::vector<float> extractLandmarks(const ncnn::Mat& face, const ROI& roi);
    
    // Internal implementation details
    void* modelHandle;
    bool isInitialized;
    int numThreads;
    mutable std::mutex mutex;
    
    // NCNN-specific models
    ncnn::Net faceDetector;
    ncnn::Net landmarkDetector;
    
    // Model parameters
    int faceDetectorWidth;
    int faceDetectorHeight;
    int landmarkDetectorWidth;
    int landmarkDetectorHeight;
};

#endif // FACE_LANDMARK_H
