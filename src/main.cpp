#include <iostream>
#include <opencv2/opencv.hpp>
#include "face_landmark.h"

int main() {
    std::cout << "Custom Face Landmark Detection" << std::endl;
    
    // Create detector
    FaceLandmarkDetector detector;
    detector.initialize();
    
    // Attempt to load model
    bool loaded = detector.loadModel("/models");
    std::cout << "Model loading " << (loaded ? "successful" : "failed") << std::endl;
    
    if (loaded) {
        // Try to process a test image if available
        try {
            cv::Mat image = cv::imread("models/test.jpg");
            if (!image.empty()) {
                std::cout << "Processing test image..." << std::endl;
                std::vector<float> landmarks = detector.detectLandmarksFromMat(image);
                
                std::cout << "Detected " << landmarks.size() / 2 << " landmarks" << std::endl;
                
                // Draw landmarks on image
                for (size_t i = 0; i < landmarks.size(); i += 2) {
                    cv::circle(image, cv::Point(landmarks[i], landmarks[i+1]), 3, cv::Scalar(0, 255, 0), -1);
                }
                
                // Save output
                cv::imwrite("landmark_output.jpg", image);
                std::cout << "Output saved to landmark_output.jpg" << std::endl;
            } else {
                std::cout << "No test image found" << std::endl;
            }
        } catch (const cv::Exception& e) {
            std::cerr << "OpenCV error: " << e.what() << std::endl;
        }
    }
    
    std::cout << "Testing complete!" << std::endl;
    return 0;
}
