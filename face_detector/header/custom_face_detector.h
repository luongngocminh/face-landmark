/*
 * GPUPixel
 *
 * Created by PixPark on 2023/10/30.
 * Copyright © 2023 PixPark. All rights reserved.
 */

#pragma once

#include <memory>
#include <vector>
#include "gpupixel/gpupixel_define.h"

// Forward declaration
class FaceLandmarkDetector;

namespace gpupixel {

/**
 * Custom face detector class for WebAssembly builds
 * Uses custom-face-landmark library instead of mars-face-kit
 */
class CustomFaceDetector {
 public:
  /**
   * Create a new CustomFaceDetector instance
   * @return A shared pointer to a new CustomFaceDetector instance
   */
  static std::shared_ptr<CustomFaceDetector> Create(int num_threads);

  /**
   * Default constructor
   */
  CustomFaceDetector(int num_threads);

  /**
   * Destructor
   */
  ~CustomFaceDetector();

  /**
   * Detect facial landmarks in an image
   * @param data Image data
   * @param width Image width
   * @param height Image height
   * @param stride Image stride (bytes per row)
   * @param fmt Image format (GPUPIXEL_MODE_FMT)
   * @param frameType Frame type (GPUPIXEL_FRAME_TYPE)
   * @return Vector of facial landmark coordinates (x,y pairs)
   */
  std::vector<float> Detect(const uint8_t* data, int width, int height, int stride,
                           GPUPIXEL_MODE_FMT fmt, GPUPIXEL_FRAME_TYPE frameType);
  
  /**
   * Set the number of threads for detection
   * @param num_threads Number of threads to use
   */
  void SetNumThreads(int num_threads);
  /**
   * Get the number of threads used for detection
   * @return Number of threads
   */
  int GetNumThreads() const;

 private:
  // Pointer to the custom face landmark detector implementation
  std::unique_ptr<FaceLandmarkDetector> detector_;
  // Number of threads for detection
  int num_threads_;
};

}  // namespace gpupixel
