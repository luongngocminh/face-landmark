# Custom Face Landmark Detection WebAssembly Library

A static library for detecting facial landmarks, compiled to WebAssembly for use in web applications. Supports multi-threading for improved performance.

## Dependencies

### Required
- CMake 3.10+
- C++17 compatible compiler
- Emscripten SDK
- NCNN library for neural network inference (included as a third-party dependency)

## Third-Party Libraries

This project includes the following third-party libraries:

### NCNN
NCNN is a high-performance neural network inference framework developed by Tencent. It is optimized for mobile platforms and supports various neural network architectures.

- Location: `/third-party/ncnn/`
- Website: https://github.com/Tencent/ncnn
- License: BSD 3-Clause

## Building the Project

This project is designed to be built with Emscripten for WebAssembly.

### Prerequisites

Make sure you have Emscripten installed and activated in your environment:

```bash
# Clone emsdk if you don't have it
git clone https://github.com/emscripten-core/emsdk.git

# Enter the directory
cd emsdk

# Download and install the latest SDK tools
./emsdk install latest

# Activate the latest SDK
./emsdk activate latest

# Activate path variables
source ./emsdk_env.sh
```

### Building

Use the provided build script:

```bash
# Make the script executable if needed
chmod +x build_wasm.sh

# Run the build script
./build_wasm.sh
```

Or manually build:

```bash
# Create a build directory
mkdir wasm-build && cd wasm-build

# Configure with Emscripten
emcmake cmake ..

# Build
emmake make
```

## Model Files

The library requires pre-trained model files in the `models/` directory:

- Face detector: `yoloface-500k.param` and `yoloface-500k.bin`
- Landmark detector: `landmark106.param` and `landmark106.bin`

These model files are automatically preloaded and made available to the application at runtime through Emscripten's virtual file system.

## Running the Demo

After building, you can serve the web directory from your build folder:

```bash
cd wasm-build
python3 ../demo/wasm_app/web/serve.py
```

Then open a browser and navigate to `http://localhost:8000` to see the demo.

## Threading Support

This project uses Web Workers and SharedArrayBuffer for multi-threading support in browsers. For this to work:

1. Your browser must support WebAssembly threads (most modern browsers do)
2. The web server must set proper headers:
   - `Cross-Origin-Opener-Policy: same-origin`
   - `Cross-Origin-Embedder-Policy: require-corp`

Use the provided `serve.py` script to run a properly configured server.

## Using the Library in Your Web Project

### JavaScript API

```javascript
// Initialize the module
FaceLandmarkModule().then(module => {
    // Initialize the detector
    module.ccall('initialize', 'number', [], []);
    
    // Load the model from the preloaded filesystem
    module.ccall('loadModel', 'number', ['string'], ['/models']);
    
    // Process image data (synchronous)
    const processSync = () => {
        const imageData = new Uint8Array([...]); // Your RGBA image data
        const width = 640;
        const height = 480;
        
        // Allocate memory for image data in WASM
        const imageDataPtr = module._malloc(imageData.length);
        module.HEAPU8.set(imageData, imageDataPtr);
        
        // Allocate memory for numPoints output parameter
        const numPointsPtr = module._malloc(4);
        
        // Call the detection function
        const landmarksPtr = module.ccall(
            'detectLandmarks',
            'number',
            ['number', 'number', 'number', 'number'],
            [imageDataPtr, width, height, numPointsPtr]
        );
        
        // Get the number of points detected
        const numPoints = module.getValue(numPointsPtr, 'i32');
        
        // Read landmarks
        const landmarks = [];
        for (let i = 0; i < numPoints; i++) {
            landmarks.push(module.getValue(landmarksPtr + (i * 4), 'float'));
        }
        
        // Free allocated memory
        module._free(imageDataPtr);
        module._free(numPointsPtr);
        module._free(landmarksPtr);
        
        return landmarks;
    };
    
    // For asynchronous processing (using threads)
    const processAsync = () => {
        // Set up callback function
        window._landmarkDetectionComplete = function(promiseId, landmarksPtr, numPoints) {
            // Process results
        };
        
        // Call async detection
        const promiseId = module.ccall('detectLandmarksAsync', 'number', 
            ['number', 'number', 'number'], 
            [imageDataPtr, width, height]);
    };
    
    // Cleanup when done
    module.ccall('cleanup', null, [], []);
});
```

## License

[Specify your license here]
