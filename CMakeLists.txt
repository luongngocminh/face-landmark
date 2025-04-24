cmake_minimum_required(VERSION 3.10)
project(custom_face_landmark VERSION 0.1.0 LANGUAGES CXX)

# Ensure we're using Emscripten
if(NOT EMSCRIPTEN)
  message(FATAL_ERROR "This project is configured for WebAssembly/Emscripten only. Please use emcmake.")
endif()

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Enable SIMD support for better performance
set(SIMD_FLAGS "-msimd128")

# Set necessary flags for Emscripten/WASM threading
# Add -pthread flag for all compilation (this enables atomics and bulk-memory)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread ${SIMD_FLAGS} -s USE_PTHREADS=1 --bind -s ENVIRONMENT='web,worker'")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -pthread ${SIMD_FLAGS} -s USE_PTHREADS=1 --bind -s ENVIRONMENT='web,worker'")

# Add forward declaration checking flag
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wunknown-pragmas")

# Include directories for the main project
include_directories(include)

# Include the third-party NCNN library headers
include_directories(${CMAKE_SOURCE_DIR}/third-party/ncnn/include/ncnn)

# Define NCNN library for Emscripten
add_library(ncnn STATIC IMPORTED)
set_target_properties(ncnn PROPERTIES
  IMPORTED_LOCATION ${CMAKE_SOURCE_DIR}/third-party/ncnn/lib/libncnn.a)

# Add debug visualization for development
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    add_definitions(-DDEBUG_VISUALIZATION)
endif()

# Add SIMD detection and support
add_definitions(-DWITH_SIMD)

# Define source files
set(LIB_SOURCES
    lib/face_landmark.cpp
    lib/face_landmark_impl.cpp
)

# Build the static library
add_library(${PROJECT_NAME} STATIC ${LIB_SOURCES})

# Link with NCNN library
target_link_libraries(${PROJECT_NAME} PRIVATE ncnn)

# WebAssembly specific settings
set(CMAKE_EXECUTABLE_SUFFIX ".js")

# WASM build options with pthread support and SIMD
# Increase thread pool size to prevent exhaustion (from 4 to 8)
set(WASM_COMPILE_FLAGS "-pthread ${SIMD_FLAGS} -s WASM=1 -s DISABLE_EXCEPTION_CATCHING=0 -s ALLOW_MEMORY_GROWTH=1 -s MODULARIZE=1 -s EXPORT_NAME=FaceLandmarkModule -s USE_PTHREADS=1 -s PTHREAD_POOL_SIZE=8 -s ENVIRONMENT='web,worker'")

# Add standard exported functions
set(WASM_LINK_FLAGS "${WASM_COMPILE_FLAGS} -s EXPORTED_FUNCTIONS=['_malloc','_free'] -s EXPORTED_RUNTIME_METHODS=['ccall','cwrap','FS','getValue','UTF8ToString','HEAPU8'] -s SINGLE_FILE=0")

# Add SIMD detection to JavaScript initialization
set(WASM_LINK_FLAGS "${WASM_LINK_FLAGS} -s ASSERTIONS=1 -s WASM_BIGINT")

# If models directory exists, add it to preload files
if(EXISTS ${CMAKE_SOURCE_DIR}/models)
  message(STATUS "Adding models directory to preload files")
  set(WASM_LINK_FLAGS "${WASM_LINK_FLAGS} --preload-file ${CMAKE_SOURCE_DIR}/models@/models")
endif()

# Add pthread pool size debugging option
set(WASM_LINK_FLAGS "${WASM_LINK_FLAGS} -s PTHREAD_POOL_SIZE_STRICT=0")

# Create the WASM demo app
add_executable(${PROJECT_NAME}_wasm demo/wasm_app/wasm_bindings.cpp)
target_link_libraries(${PROJECT_NAME}_wasm ${PROJECT_NAME} ncnn)
set_target_properties(${PROJECT_NAME}_wasm PROPERTIES LINK_FLAGS "${WASM_LINK_FLAGS}")