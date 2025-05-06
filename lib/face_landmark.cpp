#include "face_landmark_impl.h"
#include <iostream>
#include <thread>
#include <fstream>

// Define lastFaceImage here for DEBUG_VISUALIZATION
#ifdef DEBUG_VISUALIZATION
cv::Mat lastFaceImage;
#endif

// Implementation of Impl constructor
FaceLandmarkDetector::Impl::Impl() 
    : modelHandle(nullptr),
      isInitialized(false),
      numThreads(1),
      faceDetectorWidth(320),
      faceDetectorHeight(256),
      landmarkDetectorWidth(112),
      landmarkDetectorHeight(112),
      landmarkTracker(std::make_unique<LandmarkTracker>(106))
{
    // Default to using all available cores if possible
    int availableThreads = std::thread::hardware_concurrency();
    if (availableThreads > 0) {
        numThreads = availableThreads;
    }
}

// Implementation of Impl destructor
FaceLandmarkDetector::Impl::~Impl() {    
    // Cleanup NCNN models
    faceDetector.clear();
    landmarkDetector.clear();

    if (modelHandle != nullptr) {
        // Free model resources
        modelHandle = nullptr;
    }
}

// Initialize method implementation
bool FaceLandmarkDetector::Impl::initialize() {
    std::cout << "Initializing face landmark detector with " << numThreads << " threads..." << std::endl;
    isInitialized = true;
    return isInitialized;
}

// Helper function to check if a file exists
bool fileExists(const std::string &path) {
    std::ifstream f(path.c_str());
    return f.good();
}

// Load model implementation
bool FaceLandmarkDetector::Impl::loadModel(const std::string &modelPath) {
    if (!isInitialized) {
        return false;
    }

    std::cout << "Loading models from: " << modelPath << std::endl;

    try {
        // Load face detection model
        std::string faceDetectionModelPath = modelPath + "/yolo-fastest-opt.bin";
        std::string faceDetectionParamPath = modelPath + "/yolo-fastest-opt.param";

        // Check if model files exist
        if (!fileExists(faceDetectionParamPath) || !fileExists(faceDetectionModelPath)) {
            std::cerr << "Face detection model files not found at " << faceDetectionParamPath << std::endl;
            return false;
        }

        // Load models
        faceDetector.opt.num_threads = numThreads;
        int ret1 = faceDetector.load_param(faceDetectionParamPath.c_str());
        int ret2 = faceDetector.load_model(faceDetectionModelPath.c_str());

        if (ret1 != 0 || ret2 != 0) {
            std::cerr << "Failed to load face detection model: " << ret1 << ", " << ret2 << std::endl;
            return false;
        }

        // Load landmark detection model
        std::string landmarkModelPath = modelPath + "/landmark106.bin";
        std::string landmarkParamPath = modelPath + "/landmark106.param";

        // Check if model files exist
        if (!fileExists(landmarkParamPath) || !fileExists(landmarkModelPath)) {
            std::cerr << "Landmark model files not found at " << landmarkParamPath << std::endl;
            return false;
        }

        // Load models
        landmarkDetector.opt.num_threads = numThreads;
        int ret3 = landmarkDetector.load_param(landmarkParamPath.c_str());
        int ret4 = landmarkDetector.load_model(landmarkModelPath.c_str());

        if (ret3 != 0 || ret4 != 0) {
            std::cerr << "Failed to load landmark model: " << ret3 << ", " << ret4 << std::endl;
            return false;
        }

        std::cout << "Models loaded successfully" << std::endl;
        return true;
    }
    catch (const std::exception &e) {
        std::cerr << "Error loading models: " << e.what() << std::endl;
        return false;
    }
}

// Detect landmarks implementation
std::vector<float> FaceLandmarkDetector::Impl::detectLandmarks(const std::vector<unsigned char> &imageData, int width, int height) {
    // Create an NCNN Mat from the image data (RGBA to RGB)
    ncnn::Mat in = ncnn::Mat::from_pixels(imageData.data(), ncnn::Mat::PIXEL_RGBA2RGB, width, height);
    std::cout << "Input image size: " << in.w << " x " << in.h << std::endl;

    // Detect landmarks
    std::vector<float> landmarks = detectLandmarksFromMat(in);

    // Include ROI information at the beginning of landmarks for debugging
    std::vector<float> result;
    if (!landmarks.empty()) {
        std::vector<ROI> rois = extractROI(in);
        if (!rois.empty()) {
            // Apply Kalman filtering to stabilize landmarks if enabled
            std::vector<float> stabilizedLandmarks;
            if (landmarkTracker && landmarks.size() >= 106*2) { // Ensure we have at least 106 landmarks (x,y pairs)
                stabilizedLandmarks = landmarkTracker->update(landmarks);
            } else {
                stabilizedLandmarks = landmarks;
            }

            // Add ROI information at the beginning of landmarks
            result.reserve(stabilizedLandmarks.size() + 4); // ROI info (4 values) + landmarks

            // Add ROI coordinates
            result.push_back(rois[0].x1);
            result.push_back(rois[0].y1);
            result.push_back(rois[0].x2);
            result.push_back(rois[0].y2);

            // Add the stabilized landmarks
            result.insert(result.end(), stabilizedLandmarks.begin(), stabilizedLandmarks.end());
            return result;
        } else {
            // No ROIs but we have landmarks - still stabilize
            if (landmarkTracker && landmarks.size() >= 106*2) {
                return landmarkTracker->update(landmarks);
            }
        }
    }

    return landmarks.empty() ? landmarks : result;
}

bool FaceLandmarkDetector::Impl::isSIMDEnabled() const {
#ifdef WITH_SIMD
    return true;
#else
    return false;
#endif
}

void FaceLandmarkDetector::Impl::setNumThreads(int threads) {
    if (threads > 0) {
        numThreads = threads;
        std::cout << "Set number of threads to " << numThreads << std::endl;
    }
}

int FaceLandmarkDetector::Impl::getNumThreads() const {
    return numThreads;
}

// Stabilization methods implementation
void FaceLandmarkDetector::Impl::setStabilizationEnabled(bool enabled) {
    if (landmarkTracker) {
        landmarkTracker->setStabilizationEnabled(enabled);
    }
}

bool FaceLandmarkDetector::Impl::isStabilizationEnabled() const {
    return landmarkTracker ? landmarkTracker->isStabilizationEnabled() : false;
}

void FaceLandmarkDetector::Impl::setTemporalSmoothing(float factor) {
    if (landmarkTracker) {
        landmarkTracker->setTemporalSmoothing(factor);
    }
}

float FaceLandmarkDetector::Impl::getTemporalSmoothing() const {
    return landmarkTracker ? landmarkTracker->getTemporalSmoothing() : 0.8f;
}

// Now implement the public-facing methods that delegate to the Impl

FaceLandmarkDetector::FaceLandmarkDetector() : pImpl(std::make_unique<Impl>()) {}

FaceLandmarkDetector::~FaceLandmarkDetector() = default;

bool FaceLandmarkDetector::initialize() {
    return pImpl->initialize();
}

bool FaceLandmarkDetector::loadModel(const std::string &modelPath) {
    return pImpl->loadModel(modelPath);
}

std::vector<float> FaceLandmarkDetector::detectLandmarks(const std::vector<unsigned char> &imageData, int width, int height) {
    return pImpl->detectLandmarks(imageData, width, height);
}

void FaceLandmarkDetector::setNumThreads(int numThreads) {
    pImpl->setNumThreads(numThreads);
}

int FaceLandmarkDetector::getNumThreads() const {
    return pImpl->getNumThreads();
}

bool FaceLandmarkDetector::isSIMDEnabled() {
    return pImpl->isSIMDEnabled();
}

void FaceLandmarkDetector::setStabilizationEnabled(bool enabled) {
    pImpl->setStabilizationEnabled(enabled);
}

bool FaceLandmarkDetector::isStabilizationEnabled() const {
    return pImpl->isStabilizationEnabled();
}

void FaceLandmarkDetector::setTemporalSmoothing(float factor) {
    pImpl->setTemporalSmoothing(factor);
}

float FaceLandmarkDetector::getTemporalSmoothing() const {
    return pImpl->getTemporalSmoothing();
}

// New method to get the debugged face crop for visualization
extern "C" {
    EMSCRIPTEN_KEEPALIVE
    unsigned char *getLastFaceCrop(int *width, int *height) {
#ifdef DEBUG_VISUALIZATION
        if (lastFaceImage.empty()) {
            *width = 0;
            *height = 0;
            return nullptr;
        }

        *width = lastFaceImage.cols;
        *height = lastFaceImage.rows;

        // Create a copy of the data to return to JavaScript
        size_t bufferSize = lastFaceImage.cols * lastFaceImage.rows * 3;
        unsigned char *buffer = (unsigned char *)malloc(bufferSize);
        memcpy(buffer, lastFaceImage.data, bufferSize);

        return buffer;
#else
        *width = 0;
        *height = 0;
        return nullptr;
#endif
    }
}
