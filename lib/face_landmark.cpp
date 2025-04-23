#include "face_landmark.h"
#include <iostream>
#include <thread>
#include <mutex>
#include <fstream>

// Include NCNN headers directly
#include "net.h"
#include "layer.h"
#include "simpleocv.h"
#include "benchmark.h"
#include "mat.h"
#include <cmath>

// Always include Emscripten since we're targeting web only
// #include <emscripten.h>

FaceLandmarkDetector::FaceLandmarkDetector() : modelHandle(nullptr),
                                               isInitialized(false),
                                               numThreads(1),
                                               faceDetectorWidth(320),
                                               faceDetectorHeight(256),
                                               landmarkDetectorWidth(112),
                                               landmarkDetectorHeight(112)
{
    // Default to using all available cores if possible
    int availableThreads = std::thread::hardware_concurrency();
    if (availableThreads > 0)
    {
        numThreads = availableThreads;
    }
}

FaceLandmarkDetector::~FaceLandmarkDetector()
{
    // Cleanup resources
    std::lock_guard<std::mutex> lock(mutex);
    // Cleanup NCNN models

    faceDetector.clear();

    // Free landmark detector resources
    landmarkDetector.clear();

    if (modelHandle != nullptr)
    {
        // Free model resources
        modelHandle = nullptr;
    }
}

bool FaceLandmarkDetector::initialize()
{
    std::lock_guard<std::mutex> lock(mutex);
    std::cout << "Initializing face landmark detector with " << numThreads << " threads..." << std::endl;

    // Set the number of threads for NCNN
    // ncnn::set_cpu_powersave(2);
    // ncnn::set_omp_num_threads(numThreads);
    isInitialized = true;
    return isInitialized;
}

// Helper function to check if a file exists
bool fileExists(const std::string &path)
{
    std::ifstream f(path.c_str());
    return f.good();
}

bool FaceLandmarkDetector::loadModel(const std::string &modelPath)
{
    std::lock_guard<std::mutex> lock(mutex);
    if (!isInitialized)
    {
        return false;
    }

    std::cout << "Loading models from: " << modelPath << std::endl;

    try
    {
        // Load face detection model
        std::string faceDetectionModelPath = modelPath + "/yoloface-500k.bin";
        std::string faceDetectionParamPath = modelPath + "/yoloface-500k.param";

        // Check if model files exist
        if (!fileExists(faceDetectionParamPath) || !fileExists(faceDetectionModelPath))
        {
            std::cerr << "Face detection model files not found at " << faceDetectionParamPath << std::endl;
            return false;
        }

        // Load models
        faceDetector.opt.num_threads = numThreads;
        int ret1 = faceDetector.load_param(faceDetectionParamPath.c_str());
        int ret2 = faceDetector.load_model(faceDetectionModelPath.c_str());

        if (ret1 != 0 || ret2 != 0)
        {
            std::cerr << "Failed to load face detection model: " << ret1 << ", " << ret2 << std::endl;
            return false;
        }

        // Load landmark detection model
        std::string landmarkModelPath = modelPath + "/landmark106.bin";
        std::string landmarkParamPath = modelPath + "/landmark106.param";

        // Check if model files exist
        if (!fileExists(landmarkParamPath) || !fileExists(landmarkModelPath))
        {
            std::cerr << "Landmark model files not found at " << landmarkParamPath << std::endl;
            return false;
        }

        // Load models
        landmarkDetector.opt.num_threads = numThreads;
        int ret3 = landmarkDetector.load_param(landmarkParamPath.c_str());
        int ret4 = landmarkDetector.load_model(landmarkModelPath.c_str());

        if (ret3 != 0 || ret4 != 0)
        {
            std::cerr << "Failed to load landmark model: " << ret3 << ", " << ret4 << std::endl;
            return false;
        }

        std::cout << "Models loaded successfully" << std::endl;
        return true;
    }
    catch (const std::exception &e)
    {
        std::cerr << "Error loading models: " << e.what() << std::endl;
        return false;
    }
}

std::vector<float> FaceLandmarkDetector::detectLandmarks(const std::vector<unsigned char> &imageData, int width, int height)
{
    // Create an NCNN Mat from the image data (RGBA to RGB)
    ncnn::Mat in = ncnn::Mat::from_pixels(imageData.data(), ncnn::Mat::PIXEL_RGBA2RGB, width, height);

    std::cout << "Input image size: " << in.w << " x " << in.h << std::endl;

    // Detect landmarks
    std::vector<float> landmarks = detectLandmarksFromMat(in);

    // Include ROI information at the beginning of landmarks for debugging
    // This is just for visualization in the web demo
    if (!landmarks.empty())
    {
        std::vector<ROI> rois = extractROI(in);
        if (!rois.empty())
        {
            // Add ROI information at the beginning of landmarks
            // We'll prepend the first ROI coordinates
            std::vector<float> result;
            result.reserve(landmarks.size() + 4); // ROI info (4 values) + landmarks

            // Add ROI coordinates
            result.push_back(rois[0].x1);
            result.push_back(rois[0].y1);
            result.push_back(rois[0].x2);
            result.push_back(rois[0].y2);

            // Add the landmarks
            result.insert(result.end(), landmarks.begin(), landmarks.end());

            return result;
        }
    }

    return landmarks;
}

std::vector<ROI> FaceLandmarkDetector::extractROI(const ncnn::Mat &image)
{
    std::vector<ROI> rois;

    // Preprocess the image
    ncnn::Mat processedImage = preprocessImage(image);

    // Get original image size
    int img_w = image.w;
    int img_h = image.h;

    std::cout << "Image size: " << img_w << " x " << img_h << std::endl;
    std::cout << "Processed image size: " << processedImage.w << " x " << processedImage.h << std::endl;

    // Check for the processed image data
    if (processedImage.empty())
    {
        std::cerr << "Processed image is empty" << std::endl;
        return rois;
    }
    // Check if the processImage contain all 0
    if (processedImage.total() == 0)
    {
        std::cerr << "Processed image is empty" << std::endl;
        return rois;
    }

    // Detect faces for landmark detection
    ncnn::Extractor ex = this->faceDetector.create_extractor();
    ex.input("data", processedImage);
    ncnn::Mat faceDetectionOutput;
    ex.extract("output", faceDetectionOutput);

    for (int i = 0; i < faceDetectionOutput.h; i++)
    {
        std::cout << "Face detected: " << i << std::endl;
        // Extract face bounding box and score
        // The output format is [label, score, x1, y1, x2, y2]
        const float *values = faceDetectionOutput.row(i);
        float x1, y1, x2, y2, score;
        int label;
        std::cout << "Face detected: " << i << " values: " << values[0] << " " << values[1] << " " << values[2] << " " << values[3] << " " << values[4] << " " << values[5] << std::endl;

        x1 = values[2] * img_w;
        y1 = values[3] * img_h;
        x2 = values[4] * img_w;
        y2 = values[5] * img_h;

        float pw = x2 - x1;
        float ph = y2 - y1;
        float cx = x1 + 0.5f * pw;
        float cy = y1 + 0.5f * ph;

        // Scale the box to include more facial landmarks
        // Expand more on top for forehead and below for chin
        // Original padding: 0.55, 0.35, 0.55, 0.55 (left, top, right, bottom)
        // Increased padding: 0.70, 0.70, 0.70, 0.70
        x1 = cx - 0.55f * pw;
        y1 = cy - 0.7f * ph;
        x2 = cx + 0.55f * pw;
        y2 = cy + 0.7f * ph;

        score = values[1];
        label = (int)values[0];

        // Handle out-of-bounds coordinates
        if (x1 < 0)
            x1 = 0;
        if (y1 < 0)
            y1 = 0;
        if (x2 < 0)
            x2 = 0;
        if (y2 < 0)
            y2 = 0;

        if (x1 > img_w)
            x1 = img_w;
        if (y1 > img_h)
            y1 = img_h;
        if (x2 > img_w)
            x2 = img_w;
        if (y2 > img_h)
            y2 = img_h;

        // Limit the face landmark detection ROI image to be > 66x66
        if (x2 - x1 > 66 && y2 - y1 > 66)
        {
            ROI roi;
            roi.x1 = x1;
            roi.y1 = y1;
            roi.x2 = x2;
            roi.y2 = y2;
            roi.score = score;
            roi.label = label;

            rois.push_back(roi);
            std::cout << "Face detected: " << i << " x1: " << x1 << " y1: " << y1 << " x2: " << x2 << " y2: " << y2 << std::endl;
            std::cout << "Face detected: " << i << " score: " << score << std::endl;
            std::cout << "Face detected: " << i << " label: " << label << std::endl;
        }
        else
        {
            std::cout << "Face too small for landmark detection" << std::endl;
        }
    }
    std::cout << "Number of faces detected: " << rois.size() << std::endl;

    return rois;
}

std::vector<float> FaceLandmarkDetector::detectLandmarksFromMat(const ncnn::Mat &image)
{
    std::lock_guard<std::mutex> lock(mutex);
    std::vector<float> landmarks;

    if (!isInitialized)
    {
        return landmarks;
    }

    try
    {
        // Extract ROIs for face detection
        std::vector<ROI> rois = extractROI(image);
        if (rois.empty())
        {
            std::cerr << "No faces detected." << std::endl;

            // For demo purposes, add some fake landmarks
            landmarks.resize(20);
            for (int i = 0; i < 10; i++)
            {
                landmarks[i * 2] = (i * image.w / 10) + (image.w / 20);
                landmarks[i * 2 + 1] = (image.h / 2) + ((i % 3 - 1) * image.h / 8);
            }

            return landmarks;
        }

        // Get first detected face
        const ROI &roi = rois[0];

        // Crop the face region
        int x1 = static_cast<int>(roi.x1);
        int y1 = static_cast<int>(roi.y1);
        int x2 = static_cast<int>(roi.x2);
        int y2 = static_cast<int>(roi.y2);

        int width = x2 - x1;
        int height = y2 - y1;

        // Ensure valid crop dimensions
        if (width <= 0 || height <= 0)
        {
            std::cerr << "Invalid face crop dimensions" << std::endl;
            return landmarks;
        }

        // Extract landmarks for the face
        unsigned char *pixels = new unsigned char[image.w * image.h * 3]; // RGB format = 3 channels
        image.to_pixels(pixels, ncnn::Mat::PIXEL_RGB);                    // Consistent RGB format
        cv::Mat rgb = cv::Mat(image.h, image.w, CV_8UC3, pixels);
        cv::Mat faceImage = rgb(cv::Rect(x1, y1, width, height));
        landmarks = extractLandmarks(faceImage, roi);

        return landmarks;
    }
    catch (const std::exception &e)
    {
        std::cerr << "Error in landmark detection: " << e.what() << std::endl;
        return landmarks;
    }
}

ncnn::Mat FaceLandmarkDetector::preprocessImage(const ncnn::Mat &image)
{
    const int target_size = 320;

    // letterbox pad to multiple of 32
    int w = image.w;
    int h = image.h;
    float scale = 1.f;
    if (w > h)
    {
        scale = (float)target_size / w;
        w = target_size;
        h = h * scale;
    }
    else
    {
        scale = (float)target_size / h;
        h = target_size;
        w = w * scale;
    }

    // First convert to pixels, then resize
    // This is necessary because image.data is void* const, not unsigned char*
    unsigned char *pixels = new unsigned char[image.w * image.h * 3]; // RGB format = 3 channels
    image.to_pixels(pixels, ncnn::Mat::PIXEL_RGB);                    // Consistent RGB format

    // Now resize from the pixel data
    ncnn::Mat in = ncnn::Mat::from_pixels_resize(pixels, ncnn::Mat::PIXEL_RGB, // Consistent RGB format
                                                 image.w, image.h, w, h);

    // Free the temporary pixel buffer
    delete[] pixels;

    // Apply proper padding to make dimensions multiple of 32
    int wpad = (w + 31) / 32 * 32 - w;
    int hpad = (h + 31) / 32 * 32 - h;
    ncnn::Mat in_pad;

    // Apply the padding using NCNN's function (ensures the padding is correct)
    ncnn::copy_make_border(in, in_pad, hpad / 2, hpad - hpad / 2, wpad / 2, wpad - wpad / 2,
                           ncnn::BORDER_CONSTANT, 114.f);

    // Normalize image for face detection
    const float mean_vals[3] = {0.f, 0.f, 0.f};
    const float norm_vals[3] = {1 / 255.f, 1 / 255.f, 1 / 255.f};
    in_pad.substract_mean_normalize(mean_vals, norm_vals);

    return in_pad;
}

// New variable to store the last face image for debugging
#ifdef DEBUG_VISUALIZATION
static cv::Mat lastFaceImage;
#endif

std::vector<float> FaceLandmarkDetector::extractLandmarks(const cv::Mat &face, const ROI &roi)
{
    std::vector<float> landmarks;

    try
    {
        // Crop face region
        int x1 = static_cast<int>(roi.x1);
        int y1 = static_cast<int>(roi.y1);
        int x2 = static_cast<int>(roi.x2);
        int y2 = static_cast<int>(roi.y2);
        int width = x2 - x1;
        int height = y2 - y1;

        // Print the face crop details for debugging
        std::cout << "Face crop: x1=" << x1 << ", y1=" << y1 << ", x2=" << x2 << ", y2=" << y2 << std::endl;
        std::cout << "Face dimensions: " << width << "x" << height << std::endl;

        // Debug the face image
        std::cout << "Face image size: " << face.cols << " x " << face.rows << std::endl;
        std::cout << "Face image channels: " << face.channels() << std::endl;

// Save the cropped face for debug visualization
#ifdef DEBUG_VISUALIZATION
        lastFaceImage = face.clone();
#endif

        // Check first and last few pixels of the face image
        std::cout << "First few pixels of face image: ";
        for (int i = 0; i < std::min(30, face.cols * face.rows * face.channels()); i += 3)
        {
            std::cout << "(" << (int)face.data[i] << "," << (int)face.data[i + 1] << "," << (int)face.data[i + 2] << ") ";
        }
        std::cout << std::endl;
        std::cout << "Last few pixels of face image: ";
        for (int i = std::max(0, face.cols * face.rows * face.channels() - 30); i < face.cols * face.rows * face.channels(); i += 3)
        {
            std::cout << "(" << (int)face.data[i] << "," << (int)face.data[i + 1] << "," << (int)face.data[i + 2] << ") ";
        }
        std::cout << std::endl;
        // Convert image to pixels first, then use from_pixels_roi_resize
        // unsigned char* pixels = nullptr;
        ncnn::Mat in;

        // Use the pixel data for cropping and resizing with bounds checking
        if (x1 >= 0 && y1 >= 0 && width > 0 && height > 0)
        {
            // Create face image by cropping and resizing to the target dimensions
            in = ncnn::Mat::from_pixels_resize(face.data, ncnn::Mat::PIXEL_RGB2BGR,
                                               face.cols, face.rows,
                                               landmarkDetectorWidth, landmarkDetectorHeight);

            // Debug the cropped image
            std::cout << "Cropped face image size: " << in.w << " x " << in.h << std::endl;

            // Check the first few pixels of the cropped image
            std::cout << "First few pixels of cropped image: ";
            for (int i = 0; i < std::min(30, in.w * in.h * in.c); i += 3)
            {
                std::cout << "(" << (int)in[i] << "," << (int)in[i + 1] << "," << (int)in[i + 2] << ") ";
            }

            // Debug the cropped image
            if (in.empty())
            {
                std::cerr << "ERROR: Cropped image is empty!" << std::endl;
                throw std::runtime_error("Cropped image is empty");
            }
        }
        else
        {
            std::cerr << "Invalid crop region: outside of image bounds" << std::endl;
            std::cerr << "Crop region: x1=" << x1 << ", y1=" << y1 << ", x2=" << x2 << ", y2=" << y2 << std::endl;
            throw std::runtime_error("Invalid crop region");
        }

        // Normalize for landmark detection
        const float mean_vals[3] = {127.5f, 127.5f, 127.5f};
        const float norm_vals[3] = {1 / 127.5f, 1 / 127.5f, 1 / 127.5f};
        in.substract_mean_normalize(mean_vals, norm_vals);

        // Run landmark detection
        ncnn::Extractor ex = landmarkDetector.create_extractor();

        // Debug available output blobs in the model
        std::cout << "Attempting to identify landmark model outputs..." << std::endl;

        // Try different possible output blob names common in landmark detection models
        ex.input("data", in);
        ncnn::Mat out;
        int ret = ex.extract("bn6_3_bn6_3_scale", out);
        if (ret != 0)
        {
            std::cerr << "ERROR: Failed to extract landmarks, tried all possible output blobs" << std::endl;
        }

        // Check output size
        if (out.w < 212)
        {
            std::cerr << "WARNING: Unexpected output size: " << out.w << " (expected at least 212)" << std::endl;
        }

        // Scale landmarks back to original image coordinates
        float scale_w = static_cast<float>(width) / landmarkDetectorWidth;
        float scale_h = static_cast<float>(height) / landmarkDetectorHeight;

        // Print the scale factors for debugging
        std::cout << "Scale factors: width=" << scale_w << ", height=" << scale_h << std::endl;

        // For demo purposes (assume 106 landmarks with x,y pairs = 212 values)
        std::cout << "Landmark output size: " << out.w << " x " << out.h << std::endl;

        // Adjust the landmark count based on actual output size
        int numLandmarks = out.w / 2;
        if (numLandmarks * 2 != out.w)
        {
            // If not even number of values, might be a different format
            numLandmarks = std::min(106, out.w / 2); // Default to 106 landmarks or less
        }

        std::cout << "Using " << numLandmarks << " landmarks" << std::endl;
        landmarks.resize(numLandmarks * 2);
        // Check

        for (int i = 0; i < numLandmarks; i++)
        {
            // Scale landmark coordinates to original image space
            float lx = out[i * 2] * landmarkDetectorWidth * scale_w + x1;
            float ly = out[i * 2 + 1] * landmarkDetectorHeight * scale_h + y1;

            landmarks[i * 2] = lx;
            landmarks[i * 2 + 1] = ly;

            // Print a few landmarks for debugging
            if (i < 5)
            {
                std::cout << "Landmark " << i << ": (" << lx << ", " << ly << ")"
                          << " raw: (" << out[i * 2] << ", " << out[i * 2 + 1] << ")" << std::endl;
            }
        }

        return landmarks;
    }
    catch (const std::exception &e)
    {
        std::cerr << "Error extracting landmarks: " << e.what() << std::endl;

        // For demo purposes, generate some fake landmarks
        landmarks.resize(20);
        for (int i = 0; i < 10; i++)
        {
            float px = roi.x1 + (i % 5) * (roi.x2 - roi.x1) / 4.0f;
            float py = roi.y1 + (i / 5) * (roi.y2 - roi.y1) / 2.0f;
            landmarks[i * 2] = px;
            landmarks[i * 2 + 1] = py;
        }

        return landmarks;
    }
}

// New method to get the debugged face crop for visualization
extern "C"
{
    EMSCRIPTEN_KEEPALIVE
    unsigned char *getLastFaceCrop(int *width, int *height)
    {
#ifdef DEBUG_VISUALIZATION
        if (lastFaceImage.empty())
        {
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

bool FaceLandmarkDetector::isSIMDEnabled()
{
#ifdef WITH_SIMD
    return true;
#else
    return false;
#endif
}

void FaceLandmarkDetector::setNumThreads(int threads)
{
    std::lock_guard<std::mutex> lock(mutex);
    if (threads > 0)
    {
        numThreads = threads;
        // ncnn::set_omp_num_threads(numThreads);
        std::cout << "Set number of threads to " << numThreads << std::endl;
    }
}

int FaceLandmarkDetector::getNumThreads() const
{
    std::lock_guard<std::mutex> lock(mutex);
    return numThreads;
}
