#ifndef FACE_LANDMARK_IMPL_H
#define FACE_LANDMARK_IMPL_H

#include "face_landmark.h"
#include "landmark_tracker.h"
#include <string>
#include <vector>
#include <mutex>

// Include NCNN headers
#include "net.h"
#include "layer.h"
#include "simpleocv.h"
#include "mat.h"

// Full definition of the Impl class
class FaceLandmarkDetector::Impl {
public:
    Impl();
    ~Impl();
    
    bool initialize();
    bool loadModel(const std::string& modelPath);
    std::vector<float> detectLandmarks(const std::vector<unsigned char>& imageData, int width, int height);
    void setNumThreads(int threads);
    int getNumThreads() const;
    bool isSIMDEnabled() const;
    
    // Landmark stabilization methods
    void setStabilizationEnabled(bool enabled);
    bool isStabilizationEnabled() const;
    void setTemporalSmoothing(float factor);
    float getTemporalSmoothing() const;
    
    // Private implementation details
private:
    ncnn::Mat preprocessImage(const ncnn::Mat& image);
    std::vector<ROI> extractROI(const ncnn::Mat& image);
    std::vector<float> extractLandmarks(const cv::Mat& face, const ROI& roi, const int originalWidth, const int originalHeight);
    std::vector<float> detectLandmarksFromMat(const ncnn::Mat& image);
    
    // Helper function to calculate ROI area
    float calculateROIArea(const ROI& roi) const;

    // NCNN-specific models
    ncnn::Net faceDetector;
    ncnn::Net landmarkDetector;
    
    // State variables
    void* modelHandle;
    bool isInitialized;
    int numThreads;
    mutable std::mutex mutex;
    
    // Model parameters
    int faceDetectorWidth;
    int faceDetectorHeight;
    int landmarkDetectorWidth;
    int landmarkDetectorHeight;
    
    // Landmark tracker for stabilization
    std::unique_ptr<LandmarkTracker> landmarkTracker;
};

#endif // FACE_LANDMARK_IMPL_H
