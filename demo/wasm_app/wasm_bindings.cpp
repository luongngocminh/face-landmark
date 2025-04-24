#include <emscripten/emscripten.h>
#include <emscripten/bind.h>
#include <emscripten/threading.h>
#include "face_landmark.h"
#include <vector>
#include <cstdint>
#include <iostream>
#include <thread>
#include <atomic>

// Forward declaration for the function defined in face_landmark.cpp
extern "C" unsigned char* getLastFaceCrop(int* width, int* height);

FaceLandmarkDetector* detector = nullptr;
std::mutex detectorMutex;
std::atomic<int> activeThreads(0);  // Track active threads

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

// Debug function to get the current face crop for visualization
EMSCRIPTEN_KEEPALIVE
unsigned char* getFaceCropForDebug(int* width, int* height) {
    #ifdef DEBUG_VISUALIZATION
    return getLastFaceCrop(width, height);
    #else
    *width = 0;
    *height = 0;
    return nullptr;
    #endif
}

// Check if SIMD is enabled in the build
EMSCRIPTEN_KEEPALIVE
int isSIMDEnabled() {
    if (detector == nullptr) {
        return 0;
    }
    return detector->isSIMDEnabled() ? 1 : 0;
}

// Get current number of active threads
EMSCRIPTEN_KEEPALIVE
int getActiveThreads() {
    return activeThreads.load();
}

// Get current pthread pool size
EMSCRIPTEN_KEEPALIVE
int getPthreadPoolSize() {
    #ifdef PTHREAD_POOL_SIZE
    return PTHREAD_POOL_SIZE;
    #else
    return 8; // Default to our new size
    #endif
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

// Wrapper function for use with Emscripten bindings that returns the face crop
unsigned char* getFaceCropForDebugWrapper(int* width, int* height) {
    #ifdef DEBUG_VISUALIZATION
    return getLastFaceCrop(width, height);
    #else
    *width = 0;
    *height = 0;
    return nullptr;
    #endif
}

int getActiveThreadsWrapper() {
    return getActiveThreads();
}

int getPthreadPoolSizeWrapper() {
    return getPthreadPoolSize();
}

// Emscripten bindings
EMSCRIPTEN_BINDINGS(face_landmark_module) {
    emscripten::function("initialize", &initializeWrapper);
    emscripten::function("loadModel", &loadModelWrapper);
    emscripten::function("cleanup", &cleanupWrapper);
    emscripten::function("isThreadingSupported", &isThreadingSupportedWrapper);
    emscripten::function("getNumHardwareThreads", &getNumHardwareThreadsWrapper);
    emscripten::function("getActiveThreads", &getActiveThreadsWrapper);
    emscripten::function("getPthreadPoolSize", &getPthreadPoolSizeWrapper);
    
    // Fix the raw pointer binding by using allow_raw_pointers policy
    emscripten::function("getFaceCropForDebug", 
        &getFaceCropForDebugWrapper, 
        emscripten::allow_raw_pointers());
    emscripten::function("isSIMDEnabled", &isSIMDEnabled);
}
