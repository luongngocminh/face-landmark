/*
 * GPUPixel
 *
 * Created by PixPark on 2023/10/30.
 * Copyright © 2023 PixPark. All rights reserved.
 */

#include "gpupixel/face_detector/custom_face_detector.h"
#include "face_landmark.h"  // From custom-face-landmark library
#include <vector>
#include <memory>

namespace gpupixel {

CustomFaceDetector::CustomFaceDetector(int num_threads): num_threads_(num_threads) {
  // Create the face landmark detector with SIMD and multithreading support
  // Use new instead of make_unique since we're using C++11
  detector_ = std::unique_ptr<FaceLandmarkDetector>(new FaceLandmarkDetector());
  
  // Initialize the detector
  if (detector_) {
    detector_->initialize();
    detector_->setNumThreads(num_threads_);
    // Try loading the model (optional based on the implementation)
    detector_->loadModel("models");
  }
}

CustomFaceDetector::~CustomFaceDetector() {
  // The unique_ptr will handle cleanup automatically
  // detector_.reset() is called implicitly
}

std::shared_ptr<CustomFaceDetector> CustomFaceDetector::Create(int num_threads) {
  return std::make_shared<CustomFaceDetector>(num_threads);
}

void CustomFaceDetector::SetNumThreads(int num_threads) {
  if (detector_) {
    detector_->setNumThreads(num_threads);
  }
}
int CustomFaceDetector::GetNumThreads() const {
  if (detector_) {
    return detector_->getNumThreads();
  }
  return 0;
}

std::vector<float> CustomFaceDetector::Detect(const uint8_t* data, int width, int height, 
                                             int stride, GPUPIXEL_MODE_FMT fmt, 
                                             GPUPIXEL_FRAME_TYPE frameType) {
  if (!data || width <= 0 || height <= 0 || !detector_) {
    return std::vector<float>();
  }

  // Convert data to vector for the custom face landmark detector
  std::vector<unsigned char> imageData(data, data + (height * stride));
  
  // Detect landmarks using the custom face landmark detector
  std::vector<float> results;
  results =  detector_->detectLandmarks(imageData, width, height);

  // Get first 4 points (x, y coordinates), this is the roi coordinates
  if (results.size() < 4) {
    return std::vector<float>();
  }
  std::vector<float> landmarks;
  float roi[] = {results[0], results[1], results[2], results[3]};

  // Create a vector to hold the normalized landmarks
  std::vector<float> normalized_landmarks;
  normalized_landmarks.reserve(results.size() - 4);
  for (size_t i = 4; i < results.size(); i += 2) {
    // Normalize the landmarks to the range [0, 1]
    normalized_landmarks.push_back(results[i] / width);
    normalized_landmarks.push_back(results[i + 1] / height);
  }

  // Add the normalized landmarks to the output vector
  // each consecutive pair of values represents a landmark (x, y)
  landmarks.insert(landmarks.end(), normalized_landmarks.begin(), normalized_landmarks.end());
  // Add additional landmarks
  // Landmark 106: Center point between points 102 and 98
  landmarks.push_back((landmarks[102 * 2] + landmarks[92 * 2]) / 2);
  landmarks.push_back((landmarks[102 * 2 + 1] + landmarks[98 * 2 + 2]) / 2);

  // Landmark 107: Center point between points 35 and 65
  landmarks.push_back((landmarks[35 * 2] + landmarks[65 * 2]) / 2);
  landmarks.push_back((landmarks[35 * 2 + 1] + landmarks[65 * 2 + 1]) / 2);

  // Landmark 108: Center point between points 70 and 40
  landmarks.push_back((landmarks[70 * 2] + landmarks[40 * 2]) / 2);
  landmarks.push_back((landmarks[70 * 2 + 1] + landmarks[40 * 2 + 1]) / 2);

  // Landmark 109: Center point between points 5 and 80
  landmarks.push_back((landmarks[5 * 2] + landmarks[80 * 2]) / 2);
  landmarks.push_back((landmarks[5 * 2 + 1] + landmarks[80 * 2 + 1]) / 2);

  // Landmark 110: Center point between points 81 and 27
  landmarks.push_back((landmarks[81 * 2] + landmarks[27 * 2]) / 2);
  landmarks.push_back((landmarks[81 * 2 + 1] + landmarks[27 * 2 + 1]) / 2);

  return landmarks;
}

}  // namespace gpupixel
