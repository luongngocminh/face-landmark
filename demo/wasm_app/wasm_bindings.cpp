#include <emscripten/emscripten.h>
#include <emscripten/bind.h>
#include <emscripten/threading.h>
#include "face_landmark.h"
#include <vector>
#include <cstdint>
#include <thread>
#include <future>
#include <mutex>

#include <iostream>

FaceLandmarkDetector* detector = nullptr;
std::mutex detectorMutex;

extern "C" {

// Initialize the detector
EMSCRIPTEN_KEEPALIVE
int initialize() {
    std::lock_guard<std::mutex> lock(detectorMutex);
    if (detector == nullptr) {
        detector = new FaceLandmarkDetector();
    }
    return detector->initialize() ? 1 : 0;
}

// Load the model
EMSCRIPTEN_KEEPALIVE
int loadModel(const char* modelPath) {
    std::lock_guard<std::mutex> lock(detectorMutex);
    if (detector == nullptr) {
        return 0;
    }
    
    return detector->loadModel(std::string(modelPath)) ? 1 : 0;
}

// Process image data and return landmarks
EMSCRIPTEN_KEEPALIVE
float* detectLandmarks(uint8_t* imageData, int width, int height, int* numPoints) {
    std::lock_guard<std::mutex> lock(detectorMutex);
    if (detector == nullptr) {
        *numPoints = 0;
        return nullptr;
    }
    
    std::vector<unsigned char> imageVector(imageData, imageData + (width * height * 4));
    std::vector<float> landmarks = detector->detectLandmarks(imageVector, width, height);
    
    *numPoints = landmarks.size();
    if (*numPoints == 0) {
        return nullptr;
    }
    
    float* result = (float*)malloc(landmarks.size() * sizeof(float));
    for (size_t i = 0; i < landmarks.size(); i++) {
        result[i] = landmarks[i];
    }
    
    return result;
}

// Async landmark detection (returns a promise ID)
EMSCRIPTEN_KEEPALIVE
int detectLandmarksAsync(uint8_t* imageData, int width, int height) {
    static int nextPromiseId = 1;
    int promiseId = nextPromiseId++;
    
    // Create a copy of the image data since the original might be freed before the thread completes
    uint8_t* imageCopy = (uint8_t*)malloc(width * height * 4);
    memcpy(imageCopy, imageData, width * height * 4);
    
    // Start a new thread for detection
    std::thread([promiseId, imageCopy, width, height]() {
        // Create image vector from copied data
        std::vector<unsigned char> imageVector(imageCopy, imageCopy + (width * height * 4));
        
        // Get landmarks (with mutex protection)
        std::vector<float> landmarks;
        {
            std::lock_guard<std::mutex> lock(detectorMutex);
            if (detector != nullptr) {
                landmarks = detector->detectLandmarks(imageVector, width, height);
            }
        }
        
        // Free the copied image data
        free(imageCopy);
        
        // Store results in a format accessible from JavaScript
        int numPoints = landmarks.size();
        float* result = nullptr;
        
        if (numPoints > 0) {
            result = (float*)malloc(numPoints * sizeof(float));
            for (int i = 0; i < numPoints; i++) {
                result[i] = landmarks[i];
            }
        }
        
        // Call JavaScript callback with the promise ID and results
        // Use a more direct approach to call the JS function
        EM_ASM({
            try {
                if (typeof Module !== 'undefined' && typeof Module.onLandmarkDetectionComplete === 'function') {
                    Module.onLandmarkDetectionComplete($0, $1, $2);
                } else if (typeof self !== 'undefined' && typeof self._landmarkDetectionComplete === 'function') {
                    self._landmarkDetectionComplete($0, $1, $2);
                } else if (typeof window !== 'undefined' && typeof window._landmarkDetectionComplete === 'function') {
                    window._landmarkDetectionComplete($0, $1, $2);
                } else {
                    console.error("Error: Cannot find landmark detection callback function in any scope");
                }
            } catch (e) {
                console.error("Error calling landmark detection callback:", e);
            }
        }, promiseId, result, numPoints);
        
        // Note: JavaScript is responsible for freeing result using freeMemory
    }).detach();
    
    return promiseId;
}

// Free memory
EMSCRIPTEN_KEEPALIVE
void freeMemory(void* ptr) {
    free(ptr);
}

// Cleanup
EMSCRIPTEN_KEEPALIVE
void cleanup() {
    std::lock_guard<std::mutex> lock(detectorMutex);
    if (detector != nullptr) {
        delete detector;
        detector = nullptr;
    }
}

// Check if threading is supported
EMSCRIPTEN_KEEPALIVE
int isThreadingSupported() {
    return emscripten_has_threading_support();
}

// Get number of hardware threads available
EMSCRIPTEN_KEEPALIVE
int getNumHardwareThreads() {
    return std::thread::hardware_concurrency();
}

} // extern "C"

// Wrapper functions for use with Emscripten bindings
int initializeWrapper() {
    return initialize();
}

int loadModelWrapper(const std::string& modelPath) {
    return loadModel(modelPath.c_str());
}

void cleanupWrapper() {
    cleanup();
}

int isThreadingSupportedWrapper() {
    return isThreadingSupported();
}

int getNumHardwareThreadsWrapper() {
    return getNumHardwareThreads();
}

// Emscripten bindings
EMSCRIPTEN_BINDINGS(face_landmark_module) {
    emscripten::function("initialize", &initializeWrapper);
    emscripten::function("loadModel", &loadModelWrapper);
    emscripten::function("cleanup", &cleanupWrapper);
    emscripten::function("isThreadingSupported", &isThreadingSupportedWrapper);
    emscripten::function("getNumHardwareThreads", &getNumHardwareThreadsWrapper);
}
