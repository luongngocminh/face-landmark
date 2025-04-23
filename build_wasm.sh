#!/bin/bash

# Create build directory
mkdir -p wasm-build
cd wasm-build

# Configure with Emscripten
echo "Configuring project with Emscripten..."
emcmake cmake .. -DWASM_FEATURE=simd-threads

# Build
echo "Building project..."
emmake make -j$(nproc)

# Copy web files to the build directory root for easier serving
echo "Copying web files to build directory..."
cp -r ../demo/wasm_app/web/* ./
cp custom_face_landmark_wasm.js ./
cp custom_face_landmark_wasm.wasm ./
cp custom_face_landmark_wasm.worker.js ./

# If there's a data file from preloading, make sure it's in the web root
if [ -f "custom_face_landmark_wasm.data" ]; then
    cp custom_face_landmark_wasm.data ./
fi

# Go back to the original directory
cd ..

echo "Build completed. The output is in the wasm-build directory."
echo ""
echo "To run the demo, navigate to the wasm-build directory and use:"
echo "  cd wasm-build"
echo "  python3 serve.py"
