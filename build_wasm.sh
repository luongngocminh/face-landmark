#!/bin/bash

# Create build directory
mkdir -p wasm-build
cd wasm-build

# Configure with Emscripten - enable SIMD features
echo "Configuring project with Emscripten (SIMD enabled)..."
emcmake cmake .. -DCMAKE_BUILD_TYPE=Release

# Build
echo "Building project..."
emmake make -j$(nproc)

# Copy web files to the build directory root for easier serving
echo "Copying web files to build directory..."
cp -r ../demo/wasm_app/web/* ./

# Go back to the original directory
cd ..

echo "Build completed. The output is in the wasm-build directory."
echo ""
echo "To run the demo, navigate to the wasm-build directory and use:"
echo "  cd wasm-build"
echo "  python3 serve.py"
